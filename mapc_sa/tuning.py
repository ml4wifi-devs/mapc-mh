"""
Hyperparameter tuning for SA versions using Optuna (TPE sampler).

For each trial, all random scenarios are run in parallel (joblib) for 2000 SA steps.
The mean best data rate across scenarios is the objective (maximized).

Usage:
    python -m mapc_sa.tuning --version 1 --n_trials 100 --n_jobs 24 --seed 42
    python -m mapc_sa.tuning --n_trials 50 --n_jobs 12  # runs all 4 versions
"""
from __future__ import annotations

import mapc_sa._env  # noqa: F401

import json
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import optuna


def _run_scenario_trial(scenario_idx: int, version: int, seed: int, n_steps: int, top_n: int, params: dict) -> tuple[float, float]:
    """Run SA on one scenario and return (best_rate, T_0). Executed in a worker process."""
    import mapc_sa._env  # noqa: F401

    from mapc_sa.scenarios import RANDOM_SCENARIOS
    from mapc_sa.versions import VERSION_RUNNERS

    scenario = RANDOM_SCENARIOS[scenario_idx]
    sub_scenario, _ = scenario.split_scenario()[0]

    result = VERSION_RUNNERS[version](
        sub_scenario,
        seed=seed + scenario_idx,
        n_steps=n_steps,
        top_n=top_n,
        **params,
    )
    return result.best_rate, result.T_0


def make_objective(version: int, seed: int, n_steps: int, top_n: int):
    """Build an Optuna objective function for the given SA version."""

    def objective(trial: optuna.Trial) -> float:
        T_decay = trial.suggest_float('T_decay', 0.99, 0.9999, log=True)
        params = {'T_decay': T_decay}

        if version >= 3:
            params['max_jump_mcs']   = trial.suggest_int('max_jump_mcs', 1, 13)
            params['max_jump_power'] = trial.suggest_int('max_jump_power', 1, 3)

        if version == 4:
            params['concurrency_target'] = trial.suggest_float('concurrency_target', 1.0, 16.0)

        from mapc_sa.scenarios import RANDOM_SCENARIOS
        n_scenarios = len(RANDOM_SCENARIOS)

        results = [_run_scenario_trial(i, version, seed, n_steps, top_n, params) 
                   for i in range(n_scenarios)]

        rates, t0_values = zip(*results)
        trial.set_user_attr('T_0', float(np.mean(t0_values)))
        return float(np.mean(rates))

    return objective


def _tune_version(version: int, args) -> None:
    output = args.output or f'results/best_params_v{version}.json'
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)

    study_name = args.study_name or f'sa_v{version}'
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = make_objective(
        version=version,
        seed=args.seed,
        n_steps=args.n_steps,
        top_n=args.top_n,
    )

    print(f'Tuning SA v{version} | {args.n_trials} trials | '
          f'{args.n_jobs} scenario jobs | {args.n_steps} steps/scenario')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
        )

    best_params = study.best_params
    best_value  = study.best_value
    best_T_0    = study.best_trial.user_attrs['T_0']

    print(f'  Best value: {best_value:.4f} Mb/s | params: {best_params} | T_0: {best_T_0:.6f}')

    result = {
        'version':     version,
        'best_value':  best_value,
        'best_params': {**best_params, 'T_0': best_T_0},
        'n_trials':    args.n_trials,
        'n_steps':     args.n_steps,
        'seed':        args.seed,
    }

    with open(output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'  Saved to {output}')


def main():
    parser = ArgumentParser(description='Hyperparameter tuning for SA versions')
    parser.add_argument('--version', type=int, default=None, choices=[1, 2, 3, 4],
                        help='SA version to tune (default: all 4 sequentially)')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Parallel workers (default: 8)')
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--top_n',   type=int, default=10)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--output',  type=str, default=None,
                        help='Output JSON path (only used when --version is specified)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g. sqlite:///tuning.db)')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Optuna study name (only used when --version is specified)')
    args = parser.parse_args()

    versions = [args.version] if args.version is not None else [1, 2, 3, 4]
    for version in versions:
        _tune_version(version, args)


if __name__ == '__main__':
    main()
