"""Plot convergence curves (best_history) from evaluate.py JSON files.

Usage:
    python -m mapc_sa.plot --input results/eval_sa.json results/eval_rrhc.json results/eval_tabu.json
    python -m mapc_sa.plot --input results/eval_*.json --output results/plots/convergence
"""
from __future__ import annotations

import mapc_sa.env  # noqa: F401

import csv
import json
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS      = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
LINE_STYLES = ['-', '--', '-.']


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


def plot_results(data_list: list[dict], labels: list[str], output_stem: str, window: int = 50):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    csv_rows: list[dict] = []

    for i, (data, label) in enumerate(zip(data_list, labels)):
        histories              = [r['best_history'] for r in data['runs']]
        steps, mean, lo95, hi95 = _compute_stats(histories, window)
        color                  = COLORS[i % len(COLORS)]
        ls                     = LINE_STYLES[i % len(LINE_STYLES)]

        ax.plot(steps, mean, color=color, linestyle=ls, linewidth=1.5, label=label)
        ax.fill_between(steps, lo95, hi95, color=color, alpha=0.15)

        for s, m, l, h in zip(steps.tolist(), mean.tolist(), lo95.tolist(), hi95.tolist()):
            csv_rows.append({'step': s, 'label': label, 'mean': m, 'lo95': l, 'hi95': h})

    ax.set_xlabel('Step')
    ax.set_ylabel('Throughput (Mb/s)')
    ax.legend(framealpha=0.9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_stem) if os.path.dirname(output_stem) else '.', exist_ok=True)
    fig.savefig(f'{output_stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{output_stem}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(f'{output_stem}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'label', 'mean', 'lo95', 'hi95'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f'Saved: {output_stem}.{{pdf,png,csv}}')


def main():
    parser = ArgumentParser(description='Plot convergence curves from evaluate.py results')
    parser.add_argument('--input',  nargs='+', required=True, help='eval JSON files')
    parser.add_argument('--labels', nargs='+', default=None,  help='Legend labels (default: method from JSON)')
    parser.add_argument('--output', type=str,  default=None,  help='Output path stem')
    parser.add_argument('--window', type=int,  default=50,    help='Moving-average smoothing window')
    args = parser.parse_args()

    data_list = []
    for p in args.input:
        with open(p) as f:
            data_list.append(json.load(f))

    labels = args.labels or [
        d.get('method', os.path.splitext(os.path.basename(p))[0])
        for d, p in zip(data_list, args.input)
    ]

    output_stem = args.output or os.path.join(
        os.path.dirname(args.input[0]) or '.', 'convergence',
    )

    plot_results(data_list, labels, output_stem, window=args.window)


if __name__ == '__main__':
    main()
