"""
Final evaluation of SA on residential scenarios using best hyperparameters.

Supports running multiple seeds in parallel (--n_runs) using joblib.

Usage:
    python -m mapc_sa.evaluate --version 1 --params results/best_params_v1.json
    python -m mapc_sa.evaluate --params results/best_params_v{}.json  # all 4 versions
    python -m mapc_sa.evaluate --version 1 --params results/best_params_v1.json \\
        --n_steps 2000 --top_n 10 --n_runs 4 --n_jobs 4 --output results/eval_v1.json
"""
from __future__ import annotations

import mapc_sa._env  # noqa: F401

import json
import os
import time
from argparse import ArgumentParser

from joblib import Parallel, delayed
from tqdm import tqdm

from mapc_sa.config import config_to_serializable
from mapc_sa.scenarios import RESIDENTIAL_SCENARIOS
from mapc_sa.versions import VERSION_RUNNERS


def _run_single(scenario_idx: int, split_idx: int, version: int,
                params: dict, n_steps: int, top_n: int, seed: int) -> dict:
    """Run one SA replicate on one scenario split. Executed in a worker process."""
    import mapc_sa._env  # noqa: F401

    from mapc_sa.scenarios import RESIDENTIAL_SCENARIOS
    from mapc_sa.versions import VERSION_RUNNERS
    from mapc_sa.config import config_to_serializable

    scenario = RESIDENTIAL_SCENARIOS[scenario_idx]
    sub_scenario, _ = scenario.split_scenario()[split_idx]

    result = VERSION_RUNNERS[version](
        sub_scenario,
        seed=seed,
        n_steps=n_steps,
        top_n=top_n,
        **params,
    )

    return {
        'seed':        seed,
        'T_0':         result.T_0,
        'best_rate':   result.best_rate,
        'best_config': config_to_serializable(result.best_config, result.info),
        'top_configs': [
            {'rate': rate, 'config': config_to_serializable(cfg, result.info)}
            for rate, cfg in result.top_configs
        ],
        'history': result.history,
    }


def run_evaluation(
    version: int,
    params:  dict,
    n_steps: int,
    top_n:   int,
    seed:    int,
    n_runs:  int = 1,
    n_jobs:  int = 1,
) -> list[dict]:
    """Run SA on all residential scenarios, n_runs replicates each, n_jobs in parallel."""
    all_results = []

    for scenario_idx, scenario in enumerate(tqdm(RESIDENTIAL_SCENARIOS, desc='Scenarios')):
        splits = scenario.split_scenario()
        scenario_results = []

        for split_idx in range(len(splits)):
            seeds      = [seed + run_i for run_i in range(n_runs)]
            n_jobs_eff = min(n_jobs, n_runs)

            if n_runs == 1 or n_jobs_eff == 1:
                runs = [_run_single(scenario_idx, split_idx, version, params,
                                    n_steps, top_n, s) for s in seeds]
            else:
                runs = Parallel(n_jobs=n_jobs_eff, backend='loky')(
                    delayed(_run_single)(scenario_idx, split_idx, version, params,
                                        n_steps, top_n, s)
                    for s in seeds
                )

            scenario_results.append({'split_idx': split_idx, 'runs': list(runs)})

        all_results.append({'scenario_idx': scenario_idx, 'splits': scenario_results})

    return all_results


def _evaluate_version(version: int, args) -> None:
    params_path = args.params.format(version) if '{}' in (args.params or '') else args.params
    output      = args.output or f'results/eval_v{version}.json'
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)

    with open(params_path) as f:
        params_data = json.load(f)

    best_params   = params_data.get('best_params', params_data)
    sa_param_keys = {'T_decay', 'T_0', 'max_jump_mcs', 'max_jump_power', 'concurrency_target'}
    sa_params     = {k: v for k, v in best_params.items() if k in sa_param_keys}

    print(f'SA v{version} | params: {sa_params}')
    print(f'n_steps={args.n_steps}, top_n={args.top_n}, n_runs={args.n_runs}, '
          f'n_jobs={args.n_jobs}, seed={args.seed}')

    t0      = time.perf_counter()
    results = run_evaluation(
        version = version,
        params  = sa_params,
        n_steps = args.n_steps,
        top_n   = args.top_n,
        seed    = args.seed,
        n_runs  = args.n_runs,
        n_jobs  = args.n_jobs,
    )
    elapsed = time.perf_counter() - t0

    output_data = {
        'version':         version,
        'hyperparameters': sa_params,
        'n_steps':         args.n_steps,
        'top_n':           args.top_n,
        'n_runs':          args.n_runs,
        'seed':            args.seed,
        'elapsed_seconds': elapsed,
        'scenarios':       results,
    }

    with open(output, 'w') as f:
        json.dump(output_data, f, indent=2)

    for sc in results:
        for sp in sc['splits']:
            best_rates = [r['best_rate'] for r in sp['runs']]
            print(f"  Scenario {sc['scenario_idx']} split {sp['split_idx']}: "
                  f"best={max(best_rates):.2f} Mb/s "
                  f"(mean {sum(best_rates)/len(best_rates):.2f})")

    print(f'Total time: {elapsed:.1f}s | Saved to {output}')


def main():
    parser = ArgumentParser(description='Evaluate SA on residential scenarios')
    parser.add_argument('--version', type=int, default=None, choices=[1, 2, 3, 4],
                        help='SA version (default: all 4 sequentially)')
    parser.add_argument('--params',  type=str, required=True,
                        help='Path to best_params JSON from tuning.py. '
                             'Use {} as version placeholder when running all versions '
                             '(e.g. results/best_params_v{}.json)')
    parser.add_argument('--output',  type=str, default=None,
                        help='Output JSON path (only used when --version is specified)')
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--top_n',   type=int, default=10)
    parser.add_argument('--n_runs',  type=int, default=1,
                        help='Independent replicates per scenario (default: 1)')
    parser.add_argument('--n_jobs',  type=int, default=1,
                        help='Parallel workers for replicates (default: 1, max 24)')
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    args.n_jobs = min(args.n_jobs, 24)

    versions = [args.version] if args.version is not None else [1, 2, 3, 4]
    for version in versions:
        _evaluate_version(version, args)


if __name__ == '__main__':
    main()
