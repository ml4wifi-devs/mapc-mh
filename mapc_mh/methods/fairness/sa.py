"""Simulated Annealing for F-Optimal (F-SA) with LP-computed optimal weights.

The metaheuristic proposes sets of Co-SR sub-configurations. At each SA step,
a candidate config set is evaluated via jax.vmap (JIT-compiled), then a max-min
LP assigns optimal time fractions. The acceptance criterion compares LP-optimal
min-station throughputs (simple scalar delta, no leximin needed).

The main loop runs in Python (required because LP cannot be JIT-compiled);
config evaluation (jax.vmap) and neighbourhood generation remain JIT-compiled.
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from mapc_mh.config import ScenarioInfo
from mapc_mh.methods.fairness.core import (
    FSolution, FResult,
    make_coverage_check, make_neighbor_f, make_random_f_solution, setup_f,
    jains_index,
)
from mapc_mh.methods.fairness.weights import solve_max_min


def _calibrate_T0(
    evaluate,
    neighbor_f,
    initial_sol:       FSolution,
    initial_sub_rates: np.ndarray,
    initial_active:    np.ndarray,
    seed:              int   = 42,
    n_steps:           int   = 100,
    target_acceptance: float = 0.80,
) -> float:
    """Binary-search T_0 that gives ~target_acceptance on n_steps probe steps."""
    key = jax.random.PRNGKey(seed + 1)   # distinct from main run seed

    current_sol = initial_sol
    _, current_min_thr, _ = solve_max_min(initial_sub_rates, initial_active)

    deltas: list[float] = []
    for _ in range(n_steps):
        key, nbr_key, eval_key = jax.random.split(key, 3)
        candidate     = neighbor_f(current_sol, nbr_key)
        sub_rates_np  = np.array(evaluate(candidate, eval_key))
        active_np     = np.array(candidate.active)
        _, new_min_thr, _ = solve_max_min(sub_rates_np, active_np)
        deltas.append(new_min_thr - current_min_thr)
        current_sol     = candidate
        current_min_thr = new_min_thr

    negative_deltas = [d for d in deltas if d < 0]
    n_total     = len(deltas)
    n_improving = n_total - len(negative_deltas)

    if not negative_deltas:
        return 1.0

    T_low, T_high, T_mid = 1e-8, 1e6, 1.0
    for _ in range(100):
        T_mid    = (T_low + T_high) / 2.0
        acc      = sum(np.exp(np.clip(d / T_mid, -500.0, 0.0)) for d in negative_deltas)
        mean_acc = (n_improving + acc) / n_total
        if mean_acc < target_acceptance:
            T_low  = T_mid
        else:
            T_high = T_mid

    return float(T_mid)


def run(
    scenario,
    *,
    seed:        int         = 42,
    n_steps:     int         = 2000,
    top_n:       int         = 10,    # accepted for interface parity; unused by F-SA
    T_decay:     float       = 0.999,
    T_0:         float | None = None,
    max_configs: int         = 32,
) -> FResult:
    """Run F-SA: SA proposes config sets, LP assigns optimal weights at each step.

    Parameters
    ----------
    scenario     : simulator callable (StaticScenario)
    seed         : RNG seed
    n_steps      : number of SA iterations (Python loop; LP called each step)
    T_decay      : multiplicative temperature decay per step
    T_0          : initial temperature; auto-calibrated if None
    max_configs  : compile-time slot count (upper bound on simultaneously active configs)
    """
    info, evaluate, run_key, initial_sol, initial_sub_rates, max_configs = setup_f(
        scenario, seed, max_configs
    )
    neighbor_f = make_neighbor_f(info, max_configs)

    # Initial LP evaluation
    initial_rates_np  = np.array(initial_sub_rates)
    initial_active_np = np.array(initial_sol.active)
    _, initial_min_thr, initial_per_sta = solve_max_min(initial_rates_np, initial_active_np)

    if T_0 is None:
        T_0 = _calibrate_T0(
            evaluate, neighbor_f,
            initial_sol, initial_rates_np, initial_active_np,
            seed=seed,
        )

    # SA state
    current_sol     = initial_sol
    current_min_thr = initial_min_thr
    current_per_sta = initial_per_sta.copy()

    best_sol     = initial_sol
    best_min_thr = initial_min_thr
    best_per_sta = initial_per_sta.copy()

    T   = T_0
    rng = np.random.default_rng(seed)

    min_history:   list[float] = []
    sum_history:   list[float] = []
    score_history: list[float] = []

    for _ in range(n_steps):
        run_key, nbr_key, eval_key = jax.random.split(run_key, 3)

        candidate      = neighbor_f(current_sol, nbr_key)
        sub_rates_cand = np.array(evaluate(candidate, eval_key))
        active_np      = np.array(candidate.active)
        _, new_min_thr, new_per_sta = solve_max_min(sub_rates_cand, active_np)

        delta      = new_min_thr - current_min_thr
        log_accept = min(0.0, delta / max(T, 1e-300))
        if np.log(rng.uniform()) < log_accept:
            current_sol     = candidate
            current_min_thr = new_min_thr
            current_per_sta = new_per_sta

        if current_min_thr > best_min_thr:
            best_sol     = current_sol
            best_min_thr = current_min_thr
            best_per_sta = current_per_sta.copy()

        T *= T_decay

        min_history.append(float(best_min_thr))
        sum_history.append(float(best_per_sta.sum()))
        score_history.append(float(best_min_thr * 1e3 + best_per_sta.sum()))

    # Re-evaluate best solution for accurate final reporting
    run_key, eval_key = jax.random.split(run_key)
    best_sub_rates = np.array(evaluate(best_sol, eval_key))
    _, best_min_final, best_per_sta_final = solve_max_min(
        best_sub_rates, np.array(best_sol.active)
    )
    if best_min_final > best_min_thr:
        best_min_thr = best_min_final
        best_per_sta = best_per_sta_final

    # Sanity check: best solution must cover all stations
    check = make_coverage_check(info)
    assert bool(check(best_sol)), (
        "F-SA returned a best solution that does not cover all stations. "
        "Check make_random_f_solution and neighbor_f."
    )

    return FResult(
        best_solution = best_sol,
        best_score    = float(best_min_thr * 1e3 + best_per_sta.sum()),
        best_min_rate = float(best_min_thr),
        best_sum_rate = float(best_per_sta.sum()),
        best_fairness = jains_index(best_per_sta),
        best_per_sta  = best_per_sta.tolist(),
        history       = score_history,
        min_history   = min_history,
        sum_history   = sum_history,
        info          = info,
    )
