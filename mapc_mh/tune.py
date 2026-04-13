"""Hyperparameter tuning for all methods using Optuna (TPE sampler).

T-Optimal family: optimises best_rate (total throughput Mb/s).
F-Optimal family: optimises best_min_rate (max-min per-station throughput Mb/s).

Usage:
    python -m mapc_mh.tune --method t_sa   --n_trials 100
    python -m mapc_mh.tune --method t_rrhc --n_trials 50
    python -m mapc_mh.tune --method t_tabu --n_trials 100
    python -m mapc_mh.tune --method f_sa   --n_trials 100
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

import json
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import optuna

from mapc_mh.methods import T_METHODS, F_METHODS
from mapc_mh.scenarios import build_scenarios

ALL_METHODS = {**T_METHODS, **F_METHODS}

# T-Optimal methods report best_rate; F-Optimal methods report best_min_rate.
_F_FAMILY = set(F_METHODS.keys())


def _search_space(method: str, trial: optuna.Trial) -> dict:
    if method == 't_sa':
        return {'T_decay': trial.suggest_float('T_decay', 0.99, 0.9999, log=True)}
    if method == 't_rrhc':
        return {'restart_threshold': trial.suggest_int('restart_threshold', 10, 500, log=True)}
    if method == 't_tabu':
        return {
            'tabu_size':    trial.suggest_int('tabu_size',    5,  100),
            'n_candidates': trial.suggest_int('n_candidates', 3,  30),
        }
    if method == 'f_sa':
        return {
            'T_decay':     trial.suggest_float('T_decay',     0.99, 0.9999, log=True),
            'max_configs': trial.suggest_int('max_configs',   4,    24),
        }
    if method == 'f_vns':
        return {
            'k_max':              trial.suggest_int('k_max',              2,  3),
            'local_search_steps': trial.suggest_int('local_search_steps', 5, 50, log=True),
            'max_configs':        trial.suggest_int('max_configs',         4, 24),
        }
    if method == 'f_cg':
        return {
            'inner_steps': trial.suggest_int('inner_steps', 200,  4000, log=True),
            'T_0':         trial.suggest_float('T_0',       0.1,  50.0, log=True),
            'T_decay':     trial.suggest_float('T_decay',   0.99, 0.9999, log=True),
            'patience':    trial.suggest_int('patience',    3,    30),
            'max_pool':    trial.suggest_int('max_pool',    16,   128, log=True),
            'lambda_mode': trial.suggest_categorical('lambda_mode', ['bottleneck', 'inverse_gap']),
        }
    raise ValueError(f'Unknown method: {method}')


def _extract_metric(result, method: str) -> float:
    """Extract the scalar to maximise during tuning (family-specific)."""
    if method in _F_FAMILY:
        return result.best_min_rate
    return result.best_rate


def _make_objective(method: str, scenarios: list, seed: int, n_steps: int, top_n: int):
    run_fn = ALL_METHODS[method]

    def objective(trial: optuna.Trial) -> float:
        hparams = _search_space(method, trial)
        metrics = [
            _extract_metric(
                run_fn(scenario, seed=seed + i, n_steps=n_steps, top_n=top_n, **hparams),
                method,
            )
            for i, scenario in enumerate(scenarios)
        ]
        return float(np.mean(metrics))

    return objective


def main():
    parser = ArgumentParser(description='Tune hyperparameters for a single method')
    parser.add_argument('--method',   required=True, choices=list(ALL_METHODS))
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--n_steps',  type=int, default=2000)
    parser.add_argument('--n_seeds',  type=int, default=1)
    parser.add_argument('--top_n',    type=int, default=10)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--output',   type=str, default=None,
                        help='Output path (default: mapc_mh/methods/{family}/configs/best_params_{method}.json)')
    parser.add_argument('--storage',  type=str, default=None,
                        help='Optuna storage URL (e.g. sqlite:///tune.db)')
    args = parser.parse_args()

    family  = 'fairness' if args.method in _F_FAMILY else 'throughput'
    output  = args.output or f'mapc_mh/methods/{family}/configs/best_params_{args.method}.json'
    scenarios = build_scenarios(args.n_seeds)
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study   = optuna.create_study(
        direction      = 'maximize',
        sampler        = sampler,
        study_name     = args.method,
        storage        = args.storage,
        load_if_exists = True,
    )

    objective = _make_objective(args.method, scenarios, args.seed, args.n_steps, args.top_n)

    metric_name = 'best_min_rate (Mb/s)' if args.method in _F_FAMILY else 'best_rate (Mb/s)'
    print(f'Tuning {args.method} | {args.n_trials} trials | {args.n_steps} steps | {len(scenarios)} scenarios')
    print(f'Objective: {metric_name}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    result = {
        'method':      args.method,
        'best_value':  study.best_value,
        'best_params': study.best_params,
        'n_trials':    args.n_trials,
        'n_steps':     args.n_steps,
        'n_seeds':     args.n_seeds,
        'seed':        args.seed,
    }

    with open(output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'Best value: {study.best_value:.4f}')
    print(f'Best params: {study.best_params}')
    print(f'Saved to {output}')


if __name__ == '__main__':
    main()
