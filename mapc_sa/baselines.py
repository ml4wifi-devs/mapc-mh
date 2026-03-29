"""
H-MAB and DCF baselines for residential test scenarios.

Usage:
    python -m mapc_sa.baselines --agent h_mab --n_reps 5 --output results/baselines
    python -m mapc_sa.baselines --agent dcf   --n_reps 8 --output results/baselines
"""
from __future__ import annotations

import csv
import json
import os
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import numpy as np
import simpy
from joblib import Parallel, delayed
from mapc_dcf.channel import Channel
from mapc_dcf.constants import TAU, DEFAULT_TX_POWER
from mapc_dcf.logger import Logger
from mapc_dcf.nodes import AccessPoint
from mapc_mab import MapcAgentFactory
from reinforced_lib.agents.mab import UCB
from tqdm import tqdm, trange

import mapc_sa._env  # noqa: F401

from mapc_sa.scenarios import RESIDENTIAL_SCENARIOS


def _to_python_dict(d):
    result = {}
    for k, v in d.items():
        if hasattr(k, 'item'):
            k = k.item()
        result[k] = np.asarray(v).tolist()
    return result


# ── H-MAB ─────────────────────────────────────────────────────────────────────

def run_h_mab_once(scenario, n_steps, key):
    agent = MapcAgentFactory(
        associations=_to_python_dict(scenario.associations),
        agent_type=UCB,
        agent_params_lvl1={'c': 1.5, 'gamma': 0.5},
        agent_params_lvl2={'c': 0.5, 'gamma': 0.5},
        agent_params_lvl3={'c': 0.2, 'gamma': 0.8},
        hierarchical=True,
        seed=int(key[0]),
    ).create_mapc_agent()

    reward = 0.0
    history = []

    for _ in trange(n_steps, desc='H-MAB steps', leave=False):
        key, step_key = jax.random.split(key)
        tx, tx_power = agent.sample(reward)
        data_rate, reward, _ = scenario(step_key, tx, tx_power, return_internals=True)
        history.append(float(data_rate))

    return history


def run_h_mab(scenario, n_steps, seed=42, n_reps=5):
    key = jax.random.PRNGKey(seed)
    all_histories = []

    for _ in trange(n_reps, desc='H-MAB reps'):
        key, rep_key = jax.random.split(key)
        all_histories.append(run_h_mab_once(scenario, n_steps, rep_key))

    return np.array(all_histories, dtype=np.float32)  # (n_reps, n_steps)


# ── DCF ───────────────────────────────────────────────────────────────────────

def run_dcf_single(key, run, scenario, sim_time, logger):
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


def run_dcf(scenario, seed, n_runs, warmup, tmp_dir):
    key = jax.random.PRNGKey(seed)
    sim_time = scenario.n_steps * TAU
    os.makedirs(tmp_dir, exist_ok=True)

    results_path = os.path.join(tmp_dir, 'residential')
    logger = Logger(sim_time, warmup, results_path)

    Parallel(n_jobs=min(n_runs, 16))(
        delayed(run_dcf_single)(k, r, scenario, sim_time, logger)
        for k, r in zip(jax.random.split(key, n_runs), range(1, n_runs + 1))
    )
    logger.shutdown({'n_runs': n_runs})

    with open(results_path + '.json') as f:
        results = json.load(f)

    return np.array(results['DataRate']['Data'], dtype=np.float32)  # (n_runs,)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _ci95(col: np.ndarray):
    mean = col.mean()
    ci   = 1.96 * col.std(ddof=1) / np.sqrt(len(col)) if len(col) > 1 else 0.0
    return float(mean), float(mean - ci), float(mean + ci)


def save_h_mab_csv(path: str, histories: np.ndarray) -> None:
    """histories: (n_reps, n_steps)"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    n_steps = histories.shape[1]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'mean', 'lo95', 'hi95'])
        for i in range(n_steps):
            mean, lo, hi = _ci95(histories[:, i])
            writer.writerow([i, mean, lo, hi])
    print(f'Saved {path}')


def save_dcf_csv(path: str, rates: np.ndarray) -> None:
    """rates: (n_runs,)"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    mean, lo, hi = _ci95(rates)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mean', 'lo95', 'hi95'])
        writer.writerow([mean, lo, hi])
    print(f'Saved {path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument('--agent',   type=str, default='h_mab', choices=['h_mab', 'dcf'])
    parser.add_argument('--n_reps',  type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--output',  type=str, default='results/baselines')
    args = parser.parse_args()

    scenario = RESIDENTIAL_SCENARIOS[0]

    if args.agent == 'h_mab':
        histories = run_h_mab(scenario, args.n_steps, args.seed, args.n_reps)
        save_h_mab_csv(os.path.join(args.output, 'h_mab.csv'), histories)

    elif args.agent == 'dcf':
        tmp_dir = os.path.join(args.output, 'dcf_tmp')
        rates = run_dcf(scenario, args.seed, args.n_reps, warmup=0.1, tmp_dir=tmp_dir)
        save_dcf_csv(os.path.join(args.output, 'dcf.csv'), rates)


if __name__ == '__main__':
    main()
