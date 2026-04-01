"""Evaluate all methods on all scenarios and save per-run histories.

Usage:
    python -m mapc_mh.evaluate
    python -m mapc_mh.evaluate --n_seeds 5 \\
                               --params_sa   mapc_mh/methods/configs/best_params_sa.json \\
                               --params_rrhc mapc_mh/methods/configs/best_params_rrhc.json \\
                               --params_tabu mapc_mh/methods/configs/best_params_tabu.json
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

import json
import os
import time
from argparse import ArgumentParser

import jax
from tqdm import tqdm

from mapc_mh.methods import METHODS, METHOD_LABELS
from mapc_mh.scenarios import build_scenarios, N_SEEDS


def _load_hparams(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params', data)


def main():
    parser = ArgumentParser(description='Evaluate all methods on all scenarios')
    parser.add_argument('--params_sa',   type=str, default='mapc_mh/methods/configs/best_params_sa.json')
    parser.add_argument('--params_rrhc', type=str, default='mapc_mh/methods/configs/best_params_rrhc.json')
    parser.add_argument('--params_tabu', type=str, default='mapc_mh/methods/configs/best_params_tabu.json')
    parser.add_argument('--output',  type=str, default='results/evaluation.json')
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--n_seeds', type=int, default=N_SEEDS,
                        help='Number of topology seeds (scenario realizations) per config')
    parser.add_argument('--n_reps',  type=int, default=1,
                        help='Number of method repetitions per scenario (different method seeds)')
    parser.add_argument('--top_n',   type=int, default=10)
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    hparams = {
        'sa':   _load_hparams(args.params_sa),
        'rrhc': _load_hparams(args.params_rrhc),
        'tabu': _load_hparams(args.params_tabu),
    }

    scenarios = build_scenarios(args.n_seeds)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    print(f'Methods: {", ".join(METHOD_LABELS[m] for m in METHODS)}')
    print(f'Scenarios: {len(scenarios)} ({len(scenarios) // args.n_seeds} configs × {args.n_seeds} seeds × {args.n_reps} reps)')
    print(f'Steps: {args.n_steps}  |  Base seed: {args.seed}')
    for m, kw in hparams.items():
        if kw:
            print(f'  {METHOD_LABELS[m]} hparams: {kw}')

    t0      = time.perf_counter()
    results = {}

    for method, run_fn in METHODS.items():
        kw   = hparams[method]
        runs = []
        for i, scenario in enumerate(tqdm(scenarios, desc=METHOD_LABELS[method])):
            for rep in range(args.n_reps):
                result = run_fn(scenario, seed=args.seed + rep, n_steps=args.n_steps, top_n=args.top_n, **kw)
                runs.append({
                    'scenario_idx': i,
                    'rep_idx':      rep,
                    'best_rate':    result.best_rate,
                    'history':      result.history,
                    'best_history': result.best_history,
                })
            jax.clear_caches()
        results[method] = runs

    elapsed = time.perf_counter() - t0
    print(f'\nFinished in {elapsed:.1f}s  ({elapsed / 60:.1f} min)')

    with open(args.output, 'w') as f:
        json.dump({
            'n_steps':         args.n_steps,
            'n_seeds':         args.n_seeds,
            'n_reps':          args.n_reps,
            'seed':            args.seed,
            'hyperparameters': hparams,
            'elapsed_seconds': elapsed,
            'results':         results,
        }, f, indent=2)

    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
