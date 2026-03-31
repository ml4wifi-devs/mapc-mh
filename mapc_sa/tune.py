"""Hyperparameter tuning for SA, RRHC, and Tabu using Optuna (TPE sampler).

Usage:
    python -m mapc_sa.tune --method sa   --n_trials 100
    python -m mapc_sa.tune --method rrhc --n_trials 50
    python -m mapc_sa.tune --method tabu --n_trials 100
"""
from __future__ import annotations

import mapc_sa.env  # noqa: F401

import json
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import optuna

from mapc_sa.methods import METHODS
from mapc_sa.scenarios import build_scenarios


def _search_space(method: str, trial: optuna.Trial) -> dict:
    if method == 'sa':
        return {'T_decay': trial.suggest_float('T_decay', 0.99, 0.9999, log=True)}
    if method == 'rrhc':
        return {'restart_threshold': trial.suggest_int('restart_threshold', 10, 500, log=True)}
    if method == 'tabu':
        return {
            'tabu_size':    trial.suggest_int('tabu_size',    5,  100),
            'n_candidates': trial.suggest_int('n_candidates', 3,  30),
        }
    raise ValueError(f'Unknown method: {method}')


def _make_objective(method: str, scenarios: list, seed: int, n_steps: int, top_n: int):
    run_fn = METHODS[method]

    def objective(trial: optuna.Trial) -> float:
        hparams = _search_space(method, trial)
        rates   = [
            run_fn(scenario, seed=seed + i, n_steps=n_steps, top_n=top_n, **hparams).best_rate
            for i, scenario in enumerate(scenarios)
        ]
        return float(np.mean(rates))

    return objective


def main():
    parser = ArgumentParser(description='Tune hyperparameters for a single method')
    parser.add_argument('--method',   required=True, choices=list(METHODS))
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--n_steps',  type=int, default=2000)
    parser.add_argument('--n_seeds',  type=int, default=1,
                        help='Scenario seeds per config (default 1; use same value as evaluate for consistency)')
    parser.add_argument('--top_n',    type=int, default=10)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--output',   type=str, default=None,
                        help='Output path (default: mapc_sa/methods/configs/best_params_{method}.json)')
    parser.add_argument('--storage',  type=str, default=None,
                        help='Optuna storage URL (e.g. sqlite:///tune.db)')
    args = parser.parse_args()

    output    = args.output or f'mapc_sa/methods/configs/best_params_{args.method}.json'
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

    print(f'Tuning {args.method} | {args.n_trials} trials | {args.n_steps} steps | {len(scenarios)} scenarios')
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

    print(f'Best value: {study.best_value:.2f} Mb/s')
    print(f'Best params: {study.best_params}')
    print(f'Saved to {output}')


if __name__ == '__main__':
    main()
