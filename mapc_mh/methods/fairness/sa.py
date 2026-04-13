"""Simulated Annealing for F-Optimal (fairness) approximation (F-SA)."""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mapc_mh.config import NetworkConfig, ScenarioInfo
from mapc_mh.methods.fairness.core import (
    FSolution, FResult,
    leximin_delta, leximin_score,
    make_neighbor_f, setup_f,
    jains_index,
)
from mapc_mh.methods.fairness.weights import WeightStrategy, identity_strategy


class _FSAState(NamedTuple):
    sol:          FSolution
    per_sta:      jax.Array    # (n_stas,) current weighted per-station rates
    score:        jax.Array    # leximin_score(per_sta) — scalar for best-tracking
    best_sol:     FSolution
    best_per_sta: jax.Array    # (n_stas,) per-station rates of best solution seen
    best_score:   jax.Array
    key:          jax.Array


def _calibrate_T0(
    scenario,
    info:              ScenarioInfo,
    evaluate,
    neighbor_f,
    max_configs:       int,
    seed:              int   = 42,
    n_steps:           int   = 200,
    target_acceptance: float = 0.80,
) -> float:
    """Binary-search for T_0 that achieves ~target_acceptance on the first n_steps.

    Uses the leximin_delta acceptance criterion (not total throughput).
    """
    from mapc_mh.methods.fairness.core import make_random_f_solution

    key           = jax.random.PRNGKey(seed)
    key, sol_key  = jax.random.split(key)
    current_sol   = make_random_f_solution(info, max_configs, sol_key)

    key, eval_key = jax.random.split(key)
    current_per_sta, _, _ = evaluate(current_sol, eval_key)

    deltas: list[float] = []
    for _ in range(n_steps):
        key, nbr_key, eval_key = jax.random.split(key, 3)
        candidate   = neighbor_f(current_sol, nbr_key)
        c_per_sta, _, _ = evaluate(candidate, eval_key)
        delta       = float(leximin_delta(c_per_sta, current_per_sta))
        deltas.append(delta)
        current_sol     = candidate
        current_per_sta = c_per_sta

    negative_deltas = [d for d in deltas if d < 0]
    n_total         = len(deltas)
    n_improving     = n_total - len(negative_deltas)

    if not negative_deltas:
        return 1.0

    T_low, T_high = 1e-8, 1e6
    for _ in range(100):
        T_mid    = (T_low + T_high) / 2.0
        acc      = sum(np.exp(np.clip(d / T_mid, -500.0, 0.0)) for d in negative_deltas)
        mean_acc = (n_improving + acc) / n_total
        if mean_acc < target_acceptance:
            T_low  = T_mid
        else:
            T_high = T_mid

    return float(T_mid)


def _make_runner(
    scenario,
    info:        ScenarioInfo,
    evaluate,
    neighbor_f,
    T_0:         float,
    T_decay:     float,
    n_steps:     int,
    max_configs: int,
):
    n_aps, max_stas = info.n_aps, info.max_stas
    T_0_f     = jnp.float32(T_0)
    T_decay_f = jnp.float32(T_decay)

    def _step(state: _FSAState, step_idx: jax.Array):
        T   = T_0_f * (T_decay_f ** step_idx.astype(jnp.float32))
        key, nbr_key, eval_key, acc_key = jax.random.split(state.key, 4)

        candidate       = neighbor_f(state.sol, nbr_key)
        per_sta, _, _   = evaluate(candidate, eval_key)

        delta      = leximin_delta(per_sta, state.per_sta)
        log_accept = jnp.minimum(jnp.float32(0.0), delta / jnp.maximum(T, jnp.float32(1e-300)))
        accept     = jnp.log(jax.random.uniform(acc_key)) < log_accept

        new_sol     = jax.lax.cond(accept, lambda: candidate,     lambda: state.sol)
        new_per_sta = jax.lax.cond(accept, lambda: per_sta,       lambda: state.per_sta)
        new_score   = jax.lax.cond(accept, lambda: leximin_score(per_sta), lambda: state.score)

        improved = leximin_delta(new_per_sta, state.best_per_sta) > jnp.float32(0.0)
        new_best_sol, new_best_per_sta, new_best_score = jax.lax.cond(
            improved,
            lambda: (new_sol, new_per_sta, new_score),
            lambda: (state.best_sol, state.best_per_sta, state.best_score),
        )
        new_state = _FSAState(
            sol          = new_sol,
            per_sta      = new_per_sta,
            score        = new_score,
            best_sol     = new_best_sol,
            best_per_sta = new_best_per_sta,
            best_score   = new_best_score,
            key          = key,
        )
        # Emit best-so-far min and sum for history
        return new_state, (jnp.min(new_best_per_sta), jnp.sum(new_best_per_sta), new_best_score)

    @jax.jit
    def _run(key, initial_sol, initial_per_sta):
        init_score = leximin_score(initial_per_sta)
        init_state = _FSAState(
            sol          = initial_sol,
            per_sta      = initial_per_sta,
            score        = init_score,
            best_sol     = initial_sol,
            best_per_sta = initial_per_sta,
            best_score   = init_score,
            key          = key,
        )
        return jax.lax.scan(_step, init_state, jnp.arange(n_steps, dtype=jnp.int32))

    def runner(key, initial_sol, initial_per_sta):
        final_state, (min_hist, sum_hist, score_hist) = _run(key, initial_sol, initial_per_sta)
        return (
            final_state.best_sol,
            final_state.best_per_sta,
            final_state.best_score,
            np.array(min_hist).tolist(),
            np.array(sum_hist).tolist(),
            np.array(score_hist).tolist(),
        )

    return runner


def run(
    scenario,
    *,
    seed:            int             = 42,
    n_steps:         int             = 2000,
    top_n:           int             = 10,   # accepted for interface parity; unused by F-SA
    T_decay:         float           = 0.999,
    T_0:             float | None    = None,
    max_configs:     int             = 32,
    weight_strategy: WeightStrategy  = None,
) -> FResult:
    """Run Simulated Annealing for F-Optimal (fairness) approximation.

    Parameters
    ----------
    scenario      : simulator callable (StaticScenario)
    seed          : RNG seed
    n_steps       : number of SA iterations (fully JIT-compiled via lax.scan)
    T_decay       : multiplicative temperature decay per step
    T_0           : initial temperature; auto-calibrated if None
    max_configs   : compile-time slot count (upper bound on sub-configs in solution)
    weight_strategy : WeightStrategy callable; defaults to identity_strategy (metaheuristic
                      controls weights). Swap in lp_strategy for LP ablation (requires
                      replacing lax.scan with a Python loop — see weights.py).
    """
    if weight_strategy is None:
        weight_strategy = identity_strategy

    info, evaluate, run_key, initial_sol, initial_per_sta, max_configs = setup_f(
        scenario, seed, max_configs, weight_strategy
    )
    neighbor_f = make_neighbor_f(info, max_configs)

    if T_0 is None:
        T_0 = _calibrate_T0(scenario, info, evaluate, neighbor_f, max_configs, seed=seed)

    runner = _make_runner(scenario, info, evaluate, neighbor_f, T_0, T_decay, n_steps, max_configs)

    best_sol, best_per_sta, best_score, min_hist, sum_hist, score_hist = runner(
        run_key, initial_sol, initial_per_sta
    )

    best_per_sta_np = np.array(best_per_sta)

    # Sanity check: best solution must cover all stations (Python-side assertion)
    from mapc_mh.methods.fairness.core import make_coverage_check
    check = make_coverage_check(info)
    assert bool(check(best_sol)), (
        "F-SA returned a best solution that does not cover all stations. "
        "This should not happen — check make_random_f_solution and neighbor_f."
    )

    return FResult(
        best_solution  = best_sol,
        best_score     = float(best_score),
        best_min_rate  = float(best_per_sta_np.min()),
        best_sum_rate  = float(best_per_sta_np.sum()),
        best_fairness  = jains_index(best_per_sta_np),
        best_per_sta   = best_per_sta_np.tolist(),
        history        = score_hist,
        min_history    = min_hist,
        sum_history    = sum_hist,
        info           = info,
    )
