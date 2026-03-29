"""
Plot throughput vs step for SA evaluation results.

Produces:
  - A throughput-vs-step plot with mean ± 95% CI (moving average smoothed)
  - A CSV with the same data for use in TikZ/pgfplots

The input JSON is produced by evaluate.py. Each scenario/split can have
multiple runs (seeds). All runs across all scenarios/splits are aggregated.

Usage:
    python -m mapc_sa.plot --input results/sweep_v1.json --output results/plots/sweep_v1
    python -m mapc_sa.plot --input r1.json r2.json --labels V1 V2 --output results/plots/compare
"""
from __future__ import annotations

import mapc_sa._env  # noqa: F401

import csv
import json
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np


def _extract_histories(data: dict) -> list[list[float]]:
    """Return list of per-run histories from evaluate.py output."""
    histories = []
    for scenario in data['scenarios']:
        for split in scenario['splits']:
            for run in split['runs']:
                histories.append(run['history'])
    return histories


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple uniform moving average. Output length equals input length."""
    kernel = np.ones(window) / window
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def compute_stats(
    histories: list[list[float]],
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean, lower_95, upper_95) after moving-average smoothing."""
    arr   = np.array(histories, dtype=np.float32)          # (n_runs, n_steps)
    steps = np.arange(arr.shape[1])

    smoothed = np.stack([moving_average(row, window) for row in arr])  # (n_runs, n_steps)

    mean = smoothed.mean(axis=0)
    if len(smoothed) > 1:
        sem = smoothed.std(axis=0, ddof=1) / np.sqrt(len(smoothed))
        ci  = 1.96 * sem
    else:
        ci = np.zeros_like(mean)
    return steps, mean, mean - ci, mean + ci


def save_csv(
    path:     str,
    datasets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    h_mab=None,
    dcf=None,
) -> None:
    """Save (label, steps, mean, lo95, hi95) datasets to a single CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    n_steps = datasets[0][1].shape[0]

    header = ['step']
    for label, *_ in datasets:
        safe = label.replace(' ', '_')
        header += [f'{safe}_mean', f'{safe}_lo95', f'{safe}_hi95']
    if h_mab is not None:
        header += ['H_MAB_mean', 'H_MAB_lo95', 'H_MAB_hi95']
    if dcf is not None:
        header += ['DCF_mean', 'DCF_lo95', 'DCF_hi95']

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_steps):
            row = [i]
            for _, steps, mean, lo, hi in datasets:
                row += [mean[i], lo[i], hi[i]]
            if h_mab is not None:
                _, h_mean, h_lo, h_hi = h_mab
                row += [h_mean[i], h_lo[i], h_hi[i]]
            if dcf is not None:
                row += list(dcf)
            writer.writerow(row)

    print(f'Saved CSV to {path}')


PLOT_PARAMS = {
    'figure.figsize': (5, 3),
    'figure.dpi': 120,
    'font.size': 9,
    'axes.linewidth': 0.6,
    'grid.alpha': 0.35,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
}

COLORS = ['#e31a1c', '#1f78b4', '#ff7f00', '#a65628', '#984ea3']

H_MAB_COLOR = '#33a02c'
DCF_COLOR   = '#6a3d9a'


def load_h_mab_csv(path: str):
    """Load h_mab.csv → (steps, mean, lo95, hi95)."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def load_dcf_csv(path: str):
    """Load dcf.csv → (mean, lo95, hi95)."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return float(data[0]), float(data[1]), float(data[2])


def plot_results(
    output_stem: str,
    datasets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    h_mab=None,
    dcf=None,
) -> None:
    """Plot throughput vs step with mean ± 95% CI bands."""
    plt.rcParams.update(PLOT_PARAMS)
    fig, ax = plt.subplots()

    for (label, steps, mean, lo, hi), color in zip(datasets, COLORS):
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.20)

    if h_mab is not None:
        h_steps, h_mean, h_lo, h_hi = h_mab
        ax.plot(h_steps, h_mean, color=H_MAB_COLOR, label='H-MAB')
        ax.fill_between(h_steps, h_lo, h_hi, color=H_MAB_COLOR, alpha=0.20)

    if dcf is not None:
        dcf_mean, dcf_lo, dcf_hi = dcf
        ax.axhline(dcf_mean, color=DCF_COLOR, linestyle='--', label='DCF')
        ax.axhspan(dcf_lo, dcf_hi, color=DCF_COLOR, alpha=0.10)

    ax.set_xlabel('Step')
    ax.set_ylabel('Throughput [Mb/s]')
    ax.legend(fontsize=7)
    ax.grid(True, axis='both')
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_stem) if os.path.dirname(output_stem) else '.', exist_ok=True)
    pdf_path = output_stem + '.pdf'
    png_path = output_stem + '.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f'Saved plot to {pdf_path} and {png_path}')


def main():
    parser = ArgumentParser(description='Plot SA throughput-vs-step results')
    parser.add_argument('--input',  nargs='+', required=True,
                        help='One or more evaluate.py JSON outputs')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='Labels for each input file (default: filename stems)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path stem (no extension). '
                             'Default: same dir as first input, same stem.')
    parser.add_argument('--window', type=int, default=50,
                        help='Moving average window size (default: 50)')
    parser.add_argument('--h_mab', type=str, default=None,
                        help='Path to H-MAB CSV from baselines.py')
    parser.add_argument('--dcf',   type=str, default=None,
                        help='Path to DCF CSV from baselines.py')
    args = parser.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.input]
    if len(labels) != len(args.input):
        parser.error('--labels must have same length as --input')

    output_stem = args.output or os.path.splitext(args.input[0])[0]

    datasets = []
    for label, path in zip(labels, args.input):
        with open(path) as f:
            data = json.load(f)
        histories = _extract_histories(data)
        n_runs    = len(histories)
        n_steps   = len(histories[0]) if histories else 0
        print(f'{label}: {n_runs} run(s), {n_steps} steps each')
        steps, mean, lo, hi = compute_stats(histories, window=args.window)
        datasets.append((label, steps, mean, lo, hi))

    h_mab_data = None
    if args.h_mab:
        h_steps, h_mean, h_lo, h_hi = load_h_mab_csv(args.h_mab)
        h_mean = moving_average(h_mean, args.window)
        h_lo   = moving_average(h_lo,   args.window)
        h_hi   = moving_average(h_hi,   args.window)
        h_mab_data = (h_steps, h_mean, h_lo, h_hi)
        print(f'H-MAB: {len(h_steps)} steps from {args.h_mab}')

    dcf_data = None
    if args.dcf:
        dcf_data = load_dcf_csv(args.dcf)
        print(f'DCF: mean={dcf_data[0]:.2f} [{dcf_data[1]:.2f}, {dcf_data[2]:.2f}] Mb/s')

    plot_results(output_stem, datasets, h_mab=h_mab_data, dcf=dcf_data)
    save_csv(output_stem + '.csv', datasets, h_mab=h_mab_data, dcf=dcf_data)


if __name__ == '__main__':
    main()
