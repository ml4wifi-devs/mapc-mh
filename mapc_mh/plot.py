"""Plot convergence curves per scenario config from evaluate.py / baselines.py results.

One subplot per scenario config (3×3 grid), one curve per method.
Search methods: mean ± 95% CI of best_history over seeds.
Baselines: horizontal line at mean best_rate over seeds.

Usage:
    python -m mapc_mh.plot --input results/evaluation.json
    python -m mapc_mh.plot --input results/evaluation.json results/baselines.json
    python -m mapc_mh.plot --input results/evaluation.json results/baselines.json \\
                           --output results/plots/convergence
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

import csv
import json
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mapc_mh.methods import ALL_LABELS
from mapc_mh.scenarios import SCENARIO_CONFIGS


COLORS      = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
LINE_STYLES = ['-', '--', '-.', ':']


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    pad    = window // 2
    padded = np.pad(x, pad, mode='edge')
    return np.convolve(padded, np.ones(window) / window, mode='valid')[:len(x)]


def _compute_stats(histories: list[list[float]], window: int):
    arr  = np.array(histories)
    sem  = arr.std(axis=0) / np.sqrt(len(arr))
    mean = _moving_average(arr.mean(axis=0), window)
    lo95 = _moving_average(arr.mean(axis=0) - 1.96 * sem, window)
    hi95 = _moving_average(arr.mean(axis=0) + 1.96 * sem, window)
    return np.arange(len(mean)), mean, lo95, hi95


def _load_results(input_paths: list[str]) -> dict[str, dict]:
    """Load and merge results from multiple files. Returns method -> {runs, n_seeds, n_reps}."""
    merged = {}
    for path in input_paths:
        with open(path) as f:
            data = json.load(f)
        n_seeds = data['n_seeds']
        n_reps  = data.get('n_reps', 1)
        for method, runs in data['results'].items():
            merged[method] = {'runs': runs, 'n_seeds': n_seeds, 'n_reps': n_reps}
    return merged


def _group_by_config(runs: list[dict], n_seeds: int, n_reps: int = 1) -> dict[str, list[dict]]:
    stride  = n_seeds * n_reps
    grouped = {}
    for cfg_idx, (x, y) in enumerate(SCENARIO_CONFIGS):
        start = cfg_idx * stride
        grouped[f'{x}x{y}'] = runs[start:start + stride]
    return grouped


def plot_results(merged: dict[str, dict], output_stem: str, window: int = 50):
    n_configs = len(SCENARIO_CONFIGS)
    n_cols    = 3
    n_rows    = (n_configs + n_cols - 1) // n_cols
    methods   = list(merged.keys())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharey=False)
    axes_flat = axes.flatten()

    csv_rows = []

    for cfg_idx, (x, y) in enumerate(SCENARIO_CONFIGS):
        ax   = axes_flat[cfg_idx]
        name = f'{x}x{y}'
        n_steps = None

        for i, method in enumerate(methods):
            color = COLORS[i % len(COLORS)]
            ls    = LINE_STYLES[i % len(LINE_STYLES)]
            label = ALL_LABELS.get(method, method)
            runs  = _group_by_config(merged[method]['runs'], merged[method]['n_seeds'], merged[method]['n_reps'])[name]
            runs  = [r for r in runs if r.get('best_rate') is not None or 'best_history' in r]

            if 'best_history' in runs[0]:
                histories        = [r['best_history'] for r in runs]
                steps, mean, lo95, hi95 = _compute_stats(histories, window)
                n_steps = len(steps)

                ax.plot(steps, mean, color=color, linestyle=ls, linewidth=1.5, label=label)
                ax.fill_between(steps, lo95, hi95, color=color, alpha=0.15)

                for s, m, l, h in zip(steps.tolist(), mean.tolist(), lo95.tolist(), hi95.tolist()):
                    csv_rows.append({'config': name, 'method': label, 'step': s, 'mean': m, 'lo95': l, 'hi95': h})
            else:
                rates = [r['best_rate'] for r in runs if r.get('best_rate') is not None]
                if not rates:
                    continue
                mean  = float(np.mean(rates))
                ax.axhline(mean, color=color, linestyle=ls, linewidth=1.5, label=label)
                if n_steps:
                    for s in range(n_steps):
                        csv_rows.append({'config': name, 'method': label, 'step': s, 'mean': mean, 'lo95': mean, 'hi95': mean})

        ax.set_title(f'{x}×{y}  ({x * y} APs)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Throughput (Mb/s)')
        ax.grid(True, linewidth=0.4, alpha=0.5)

    for ax in axes_flat[n_configs:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, 0.02), framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_stem) if os.path.dirname(output_stem) else '.', exist_ok=True)
    fig.savefig(f'{output_stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{output_stem}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(f'{output_stem}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'method', 'step', 'mean', 'lo95', 'hi95'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f'Saved: {output_stem}.{{pdf,png,csv}}')


def main():
    parser = ArgumentParser(description='Plot convergence curves per scenario config')
    parser.add_argument('--input',  nargs='+', required=True,
                        help='JSON files from evaluate.py and/or baselines.py')
    parser.add_argument('--output', type=str, default=None, help='Output path stem')
    parser.add_argument('--window', type=int, default=50,   help='Moving-average smoothing window')
    args = parser.parse_args()

    merged = _load_results(args.input)

    output_stem = args.output or os.path.join(
        os.path.dirname(args.input[0]) or '.', 'convergence',
    )

    plot_results(merged, output_stem, window=args.window)


if __name__ == '__main__':
    main()
