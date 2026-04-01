"""Run H-MAB and DCF baselines on all scenarios and save best_rate per scenario.

Usage:
    python -m mapc_mh.baselines
    python -m mapc_mh.baselines --agents h_mab dcf --n_seeds 5
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

import json
import os
import time
from argparse import ArgumentParser
from itertools import chain

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pulp as plp
import simpy
from joblib import Parallel, delayed
from mapc_dcf.channel import Channel
from mapc_dcf.constants import TAU, DEFAULT_TX_POWER
from mapc_dcf.logger import Logger
from mapc_dcf.nodes import AccessPoint
from mapc_mab import MapcAgentFactory
from mapc_optimal import OptimizationType, Solver, positions_to_path_loss
from reinforced_lib.agents.mab import UCB
from tqdm import tqdm

from mapc_mh.scenarios import build_scenarios, N_SEEDS


def _to_python_dict(d):
    result = {}
    for k, v in d.items():
        k = k.item() if hasattr(k, 'item') else k
        result[k] = np.asarray(v).tolist()
    return result


# ── H-MAB ─────────────────────────────────────────────────────────────────────

def _run_h_mab_once(scenario, n_steps: int, key: jax.Array) -> list[float]:
    agent = MapcAgentFactory(
        associations    = _to_python_dict(scenario.associations),
        agent_type      = UCB,
        agent_params_lvl1 = {'c': 1.5, 'gamma': 0.5},
        agent_params_lvl2 = {'c': 0.5, 'gamma': 0.5},
        agent_params_lvl3 = {'c': 0.2, 'gamma': 0.8},
        hierarchical    = True,
        seed            = int(key[0]),
    ).create_mapc_agent()

    reward  = 0.0
    history = []

    for _ in range(n_steps):
        key, step_key     = jax.random.split(key)
        tx, tx_power      = agent.sample(reward)
        data_rate, reward, _ = scenario(step_key, tx, tx_power, return_internals=True)
        history.append(float(data_rate))

    return history


def run_h_mab(scenario, n_steps: int, seed: int) -> dict:
    key     = jax.random.PRNGKey(seed)
    history = _run_h_mab_once(scenario, n_steps, key)
    return {
        'best_rate': float(np.mean(history)),
        'history':   history,
    }


# ── DCF ───────────────────────────────────────────────────────────────────────

def _run_dcf_single(key, run, scenario, sim_time, logger):
    key, key_channel = jax.random.split(key)
    des_env = simpy.Environment()
    channel = Channel(key_channel, False, scenario.channel_width, scenario.pos, scenario.walls)

    for ap in scenario.associations:
        key, key_ap = jax.random.split(key)
        clients = jnp.array(scenario.associations[ap])
        ap_node = AccessPoint(key_ap, ap, scenario.pos, DEFAULT_TX_POWER, clients,
                              channel, des_env, logger)
        ap_node.start_operation(run)

    des_env.run(until=(logger.warmup_length + sim_time))
    logger.dump_acumulators(run)
    del des_env


def run_dcf(scenario, n_steps: int, seed: int, n_runs: int = 5, output_dir: str = 'results/dcf') -> dict:
    key      = jax.random.PRNGKey(seed)
    sim_time = n_steps * TAU
    os.makedirs(output_dir, exist_ok=True)
    logger       = Logger(sim_time, warmup_length=0.1, results_path=output_dir)

    Parallel(n_runs)(
        delayed(_run_dcf_single)(k, r, scenario, sim_time, logger)
        for k, r in zip(jax.random.split(key, n_runs), range(1, n_runs + 1))
    )
    logger.shutdown({'n_runs': n_runs})

    df = pd.read_csv(output_dir + '.csv')
    sim_time_max = df['SimTime'].max()
    rates_per_run = df.groupby('RunNumber')['AMPDUSize'].sum() * 1e-6 / sim_time_max

    return rates_per_run.tolist()


# ── T-Optimal (SUM) ───────────────────────────────────────────────────────────

def run_t_optimal(scenario, n_steps: int, seed: int) -> dict:
    associations  = _to_python_dict(scenario.associations)
    access_points = list(associations.keys())
    stations      = list(chain.from_iterable(associations.values()))
    path_loss     = positions_to_path_loss(np.array(scenario.pos), np.array(scenario.walls))

    solver        = Solver(
        stations      = stations,
        access_points = access_points,
        channel_width = scenario.channel_width,
        opt_type      = OptimizationType.SUM,
        solver        = plp.CPLEX_CMD(msg=False, threads=None)
    )
    _, total_rate = solver(path_loss, associations)
    return {'best_rate': float(total_rate)}


# ── CLI ───────────────────────────────────────────────────────────────────────

AGENTS = {
    'h_mab':     run_h_mab,
    'dcf':       run_dcf,
    't_optimal': run_t_optimal,
}

AGENT_LABELS = {
    'h_mab':     'H-MAB',
    'dcf':       'DCF',
    't_optimal': 'T-Optimal',
}


def main():
    parser = ArgumentParser(description='Run H-MAB and DCF baselines on all scenarios')
    parser.add_argument('--agents',  nargs='+', default=list(AGENTS), choices=list(AGENTS))
    parser.add_argument('--output',  type=str,  default='results/baselines.json')
    parser.add_argument('--n_steps', type=int,  default=2000)
    parser.add_argument('--n_seeds', type=int,  default=N_SEEDS)
    parser.add_argument('--seed',    type=int,  default=42)
    parser.add_argument('--n_reps',  type=int, default=1,
                        help='Number of repetitions per scenario (different method seeds)')
    parser.add_argument('--t_optimal_max_aps', type=int, default=16,
                        help='For t_optimal: skip configs with more APs than this (default: 16 = up to 4×4)')
    parser.add_argument('--only_first_2x2', action='store_true',
                        help='For h_mab and dcf: run only on the 2×2 config')
    args = parser.parse_args()

    from mapc_mh.scenarios import SCENARIO_CONFIGS

    scenarios = build_scenarios(args.n_seeds)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    print(f'Agents: {", ".join(AGENT_LABELS[a] for a in args.agents)}')
    print(f'Scenarios: {len(scenarios)} ({len(scenarios) // args.n_seeds} configs × {args.n_seeds} seeds × {args.n_reps} reps)')
    print(f'Steps: {args.n_steps}  |  Base seed: {args.seed}')
    if 't_optimal' in args.agents:
        print(f't_optimal restricted to configs with ≤{args.t_optimal_max_aps} APs')

    t0      = time.perf_counter()
    results = {}

    for agent in args.agents:
        run_fn = AGENTS[agent]
        runs = []
        for i, scenario in enumerate(tqdm(scenarios, desc=AGENT_LABELS[agent])):
            cfg_idx = i // args.n_seeds
            x, y    = SCENARIO_CONFIGS[cfg_idx]

            skip = False
            if args.only_first_2x2 and (x, y) != (2, 2):
                skip = True
            if agent == 't_optimal' and x * y > args.t_optimal_max_aps:
                skip = True
            
            if skip:
                runs.extend([{'scenario_idx': i, 'rep_idx': rep, 'best_rate': None} for rep in range(args.n_reps)])
            elif agent == 'dcf':
                run_result = run_fn(scenario, n_steps=args.n_steps, seed=args.seed, n_runs=args.n_reps)
                for rep_idx, rate in enumerate(run_result):
                    runs.append({'scenario_idx': i, 'rep_idx': rep_idx, 'best_rate': rate})
            else:
                for rep in range(args.n_reps):
                    run_result = run_fn(scenario, n_steps=args.n_steps, seed=args.seed + rep)
                    runs.append({'scenario_idx': i, 'rep_idx': rep, **run_result})

        results[agent] = runs

    elapsed = time.perf_counter() - t0
    print(f'\nFinished in {elapsed:.1f}s  ({elapsed / 60:.1f} min)')

    with open(args.output, 'w') as f:
        json.dump({
            'n_steps':         args.n_steps,
            'n_seeds':         args.n_seeds,
            'n_reps':          args.n_reps,
            'seed':            args.seed,
            'elapsed_seconds': elapsed,
            'results':         results,
        }, f, indent=2)

    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
