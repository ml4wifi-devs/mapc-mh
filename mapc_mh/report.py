"""Statistical comparison report from saved evaluation/baseline results.

Accepts any number of JSON files produced by evaluate.py or baselines.py
and prints a comparison table plus pairwise Mann-Whitney U significance tests.

Usage:
    python -m mapc_mh.report --input results/evaluation.json
    python -m mapc_mh.report --input results/evaluation.json results/baselines.json
    python -m mapc_mh.report --input results/evaluation.json results/baselines.json --alpha 0.01
"""
from __future__ import annotations

import json
from argparse import ArgumentParser

import numpy as np
from scipy import stats

from mapc_mh.methods import ALL_LABELS
from mapc_mh.scenarios import SCENARIO_CONFIGS


def _group_rates(data: dict) -> dict[str, dict[str, list[float]]]:
    """Group best_rates by method → scenario_name → [rates]."""
    n_seeds = data['n_seeds']
    grouped: dict[str, dict[str, list[float]]] = {}

    for method, runs in data['results'].items():
        grouped[method] = {}
        for cfg_idx, (x, y) in enumerate(SCENARIO_CONFIGS):
            name  = f'{x}x{y}'
            start = cfg_idx * n_seeds
            grouped[method][name] = [runs[start + s]['best_rate'] for s in range(n_seeds)]

    return grouped


def _print_table(all_grouped: dict[str, dict[str, list[float]]], methods: list[str]) -> None:
    labels = [ALL_LABELS.get(m, m) for m in methods]
    col_w  = 17
    header = f"{'Scenario':<10}" + ''.join(f"{lbl:>{col_w}}" for lbl in labels)
    sep    = '=' * len(header)

    print('\n' + sep)
    print('BEST RATE  [mean ± std, Mb/s]')
    print(sep)
    print(header)
    print('-' * len(header))

    for x, y in SCENARIO_CONFIGS:
        name = f'{x}x{y}'
        row  = f"{name:<10}"
        for method in methods:
            rates = all_grouped[method][name]
            row  += f"{f'{np.mean(rates):.1f}±{np.std(rates):.1f}':>{col_w}}"
        print(row)

    print(sep)


def _print_pvalue_matrices(all_grouped: dict[str, dict[str, list[float]]], methods: list[str], alpha: float) -> None:
    labels = [ALL_LABELS.get(m, m) for m in methods]
    n      = len(methods)
    cell_w = 10

    print(f'\nPAIRWISE MANN-WHITNEY U  p-VALUES  (two-sided, alpha={alpha})')
    print('(*) not statistically significant\n')

    for x, y in SCENARIO_CONFIGS:
        name = f'{x}x{y}'
        print(f'Scenario {name}  ({x * y} AP{"s" if x * y > 1 else ""}):')
        print(' ' * 8 + ''.join(f"{lbl:>{cell_w}}" for lbl in labels))

        for i, lbl in enumerate(labels):
            row = f"{lbl:<8}"
            for j in range(n):
                if i == j:
                    row += f"{'---':>{cell_w}}"
                else:
                    _, p   = stats.mannwhitneyu(
                        all_grouped[methods[i]][name],
                        all_grouped[methods[j]][name],
                        alternative='two-sided',
                    )
                    marker = '' if p < alpha else '*'
                    row   += f"{f'{p:.4f}{marker}':>{cell_w}}"
            print(row)
        print()


def main():
    parser = ArgumentParser(description='Statistical comparison report from evaluation results')
    parser.add_argument('--input', nargs='+', required=True,
                        help='JSON files from evaluate.py and/or baselines.py')
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()

    all_grouped: dict[str, dict[str, list[float]]] = {}

    for path in args.input:
        with open(path) as f:
            data = json.load(f)
        all_grouped.update(_group_rates(data))
        print(f'Loaded {path}: {list(data["results"].keys())} (n_seeds={data["n_seeds"]})')

    methods = list(all_grouped.keys())
    print(f'\nMethods: {", ".join(ALL_LABELS.get(m, m) for m in methods)}')

    _print_table(all_grouped, methods)
    _print_pvalue_matrices(all_grouped, methods, alpha=args.alpha)


if __name__ == '__main__':
    main()
