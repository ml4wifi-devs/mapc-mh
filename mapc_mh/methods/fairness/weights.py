"""LP-based weight computation for F-Optimal metaheuristics.

Given a fixed set of sub-configurations and their per-station throughput rates,
solve a max-min LP to find optimal time fractions (mixing weights).

The LP (solved via scipy HiGHS — ~0.6 ms/call for typical problem sizes):
    max  t
    s.t. sum_m w_m * r[m, s] >= t   for all stations s
         sum_m w_m == 1
         w_m >= 0                   for all m
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def solve_max_min(
    sub_sta_rates: np.ndarray,
    active:        np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve a max-min LP for optimal mixing weights given a set of sub-configs.

    Parameters
    ----------
    sub_sta_rates : (max_configs, n_stas) float array — per-config per-station throughput (Mb/s)
    active        : (max_configs,) bool array — which slots are in use

    Returns
    -------
    weights   : (max_configs,) array — optimal weights; zero for inactive slots
    min_thr   : float — LP-optimal min-station throughput (Mb/s)
    per_sta   : (n_stas,) array — resulting per-station throughput
    """
    active_idx = np.where(active)[0]
    max_configs, n_stas = sub_sta_rates.shape

    if len(active_idx) == 0:
        return np.zeros(max_configs), 0.0, np.zeros(n_stas)

    rates    = sub_sta_rates[active_idx]   # (n_active, n_stas)
    n_active = len(active_idx)

    # Variables: [w_0, ..., w_{n_active-1}, t]
    # Minimize -t  (= maximize t)
    c = np.zeros(n_active + 1)
    c[-1] = -1.0

    # Inequality: t - sum_m r[m,s]*w[m] <= 0  for each station s
    A_ub = np.zeros((n_stas, n_active + 1))
    A_ub[:, :n_active] = -rates.T    # -r[m,s] coefficients for w_m
    A_ub[:, n_active]  =  1.0        # +1 coefficient for t
    b_ub = np.zeros(n_stas)

    # Equality: sum w = 1
    A_eq = np.zeros((1, n_active + 1))
    A_eq[0, :n_active] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n_active + [(0.0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    weights_full = np.zeros(max_configs)
    if res.success:
        w_active = np.maximum(res.x[:n_active], 0.0)
        total = w_active.sum()
        if total > 0:
            w_active /= total
        weights_full[active_idx] = w_active
        min_thr = max(0.0, float(-res.fun))
    else:
        # Fallback: uniform over active configs
        weights_full[active_idx] = 1.0 / n_active
        min_thr = 0.0

    per_sta = (weights_full[:, None] * sub_sta_rates).sum(axis=0)   # (n_stas,)
    return weights_full, min_thr, per_sta
