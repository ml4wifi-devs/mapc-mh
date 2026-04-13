"""Evaluate T-Optimal and F-Optimal methods on all scenarios and save per-run histories.

Usage:
    python -m mapc_mh.evaluate
    python -m mapc_mh.evaluate --skip_f \\
        --params_sa   mapc_mh/methods/throughput/configs/best_params_sa.json \\
        --params_rrhc mapc_mh/methods/throughput/configs/best_params_rrhc.json \\
        --params_tabu mapc_mh/methods/throughput/configs/best_params_tabu.json
    python -m mapc_mh.evaluate --skip_t \\
        --params_f_sa mapc_mh/methods/fairness/configs/best_params_f_sa.json
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

import json
import os
import time
from argparse import ArgumentParser

import jax
from tqdm import tqdm

from mapc_mh.methods import T_METHODS, T_METHOD_LABELS, F_METHODS, F_METHOD_LABELS
from mapc_mh.scenarios import build_scenarios, N_SEEDS


def _load_hparams(path: str | None) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params', data)


def _jains_index(per_sta: list[float]) -> float:
    """Jain's fairness index: (sum x_i)^2 / (n * sum x_i^2). Returns 1.0 for empty input."""
    if not per_sta:
        return 1.0
    s  = sum(per_sta)
    s2 = sum(x * x for x in per_sta)
    return (s * s) / (len(per_sta) * s2) if s2 > 0 else 1.0


def _serialise_t(result) -> dict:
    return {
        'best_rate':    result.best_rate,
        'history':      result.history,
        'best_history': result.best_history,
    }


def _serialise_f(result) -> dict:
    return {
        'best_min_rate': result.best_min_rate,
        'best_sum_rate': result.best_sum_rate,
        'best_fairness': result.best_fairness,
        'best_score':    result.best_score,
        'best_per_sta':  result.best_per_sta,
        'min_history':   result.min_history,
        'sum_history':   result.sum_history,
        'history':       result.history,
    }


def _run_family(
    methods:      dict,
    labels:       dict,
    hparams:      dict,
    scenarios:    list,
    n_reps:       int,
    n_steps:      int,
    top_n:        int,
    seed:         int,
    serialise_fn,
) -> dict:
    """Run one family of methods over all scenarios and return serialisable results."""
    results = {}
    for method, run_fn in methods.items():
        kw   = hparams.get(method, {})
        runs = []
        for i, scenario in enumerate(tqdm(scenarios, desc=labels[method])):
            for rep in range(n_reps):
                result = run_fn(scenario, seed=seed + rep, n_steps=n_steps, top_n=top_n, **kw)
                runs.append({
                    'scenario_idx': i,
                    'rep_idx':      rep,
                    **serialise_fn(result),
                })
            jax.clear_caches()
        results[method] = runs
    return results


def main():
    parser = ArgumentParser(description='Evaluate T-Optimal and F-Optimal methods on all scenarios')
    # T-Optimal hparam paths
    parser.add_argument('--params_sa',   type=str,
                        default='mapc_mh/methods/throughput/configs/best_params_sa.json')
    parser.add_argument('--params_rrhc', type=str,
                        default='mapc_mh/methods/throughput/configs/best_params_rrhc.json')
    parser.add_argument('--params_tabu', type=str,
                        default='mapc_mh/methods/throughput/configs/best_params_tabu.json')
    # F-Optimal hparam paths
    parser.add_argument('--params_f_sa', type=str,
                        default='mapc_mh/methods/fairness/configs/best_params_f_sa.json')
    # Shared
    parser.add_argument('--output',  type=str, default='results/evaluation.json')
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--n_seeds', type=int, default=N_SEEDS)
    parser.add_argument('--n_reps',  type=int, default=1)
    parser.add_argument('--top_n',   type=int, default=10)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--skip_t',  action='store_true', help='Skip T-Optimal methods')
    parser.add_argument('--skip_f',  action='store_true', help='Skip F-Optimal methods')
    args = parser.parse_args()

    t_hparams = {
        'sa':   _load_hparams(args.params_sa),
        'rrhc': _load_hparams(args.params_rrhc),
        'tabu': _load_hparams(args.params_tabu),
    }
    f_hparams = {
        'f_sa': _load_hparams(args.params_f_sa),
    }

    scenarios = build_scenarios(args.n_seeds)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    active_t = dict(T_METHODS) if not args.skip_t else {}
    active_f = dict(F_METHODS) if not args.skip_f else {}

    print(f'T-Optimal methods: {", ".join(T_METHOD_LABELS[m] for m in active_t) or "(skipped)"}')
    print(f'F-Optimal methods: {", ".join(F_METHOD_LABELS[m] for m in active_f) or "(skipped)"}')
    print(f'Scenarios: {len(scenarios)} ({len(scenarios) // args.n_seeds} configs × {args.n_seeds} seeds × {args.n_reps} reps)')
    print(f'Steps: {args.n_steps}  |  Base seed: {args.seed}')

    t0 = time.perf_counter()

    results_t = _run_family(
        active_t, T_METHOD_LABELS, t_hparams, scenarios,
        args.n_reps, args.n_steps, args.top_n, args.seed, _serialise_t,
    ) if active_t else {}

    results_f = _run_family(
        active_f, F_METHOD_LABELS, f_hparams, scenarios,
        args.n_reps, args.n_steps, args.top_n, args.seed, _serialise_f,
    ) if active_f else {}

    elapsed = time.perf_counter() - t0
    print(f'\nFinished in {elapsed:.1f}s  ({elapsed / 60:.1f} min)')

    with open(args.output, 'w') as f:
        json.dump({
            'n_steps':         args.n_steps,
            'n_seeds':         args.n_seeds,
            'n_reps':          args.n_reps,
            'seed':            args.seed,
            'elapsed_seconds': elapsed,
            'hyperparameters': {**t_hparams, **f_hparams},
            'results_t':       results_t,
            'results_f':       results_f,
        }, f, indent=2)

    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
