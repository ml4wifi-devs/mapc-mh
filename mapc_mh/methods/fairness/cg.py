"""F-CG: Primal-only Column Generation for max-min fairness.

Decomposition
-------------
Outer loop (this file):
    Maintain a pool P of sub-configurations with cached per-station rate vectors
    r_c. Solve the max-min LP over the pool:
        max t  s.t.  sum_c w_c r_{c,s} >= t forall s,  sum_c w_c = 1,  w >= 0.
    Identify the bottleneck set B = {s : r_bar_s <= t + eps} and build a
    "station weight" vector lambda on B (uniform by default).

Inner loop (pricing — reused throughput SA with a scalarised objective):
    Given lambda, find c* approximately maximising sum_s lambda_s * r_{c*,s}.

Admission test (primal):
    Solve the LP on P ∪ {c*}. Admit c* iff t strictly increases (> t + eps).
    After `patience` consecutive rejections, terminate.

No LP duals are needed. lambda is purely a primal heuristic over the
bottleneck set; progress is verified by re-solving the RMP.
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog

from mapc_mh.config import (
    NetworkConfig, ScenarioInfo,
    make_config_to_arrays, make_random_config, make_scenario_info,
)
from mapc_mh.methods.core import neighbor
from mapc_mh.methods.fairness.core import FResult, jains_index


# ---------------------------------------------------------------------------
# LP: max-min over a fixed pool (scipy HiGHS)
# ---------------------------------------------------------------------------

def solve_max_min_lp(
    R: np.ndarray,
    eps_feas: float = 1e-9,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve  max t s.t.  R^T w >= t 1,  1^T w = 1,  w >= 0.

    Parameters
    ----------
    R : (n_pool, n_stas) float — per-config per-station rates.

    Returns
    -------
    w       : (n_pool,) — optimal mixing weights.
    t       : float      — LP optimum (min achieved rate).
    per_sta : (n_stas,)  — w^T R  (achieved rate vector).
    """
    n_pool, n_stas = R.shape
    if n_pool == 0:
        return np.zeros(0), 0.0, np.zeros(n_stas)

    # vars: [w_0 ... w_{n-1}, t].  minimise  -t.
    c = np.zeros(n_pool + 1); c[-1] = -1.0

    # t - R^T w <= 0  per station
    A_ub = np.zeros((n_stas, n_pool + 1))
    A_ub[:, :n_pool] = -R.T
    A_ub[:, n_pool]  =  1.0
    b_ub = np.zeros(n_stas)

    A_eq = np.zeros((1, n_pool + 1)); A_eq[0, :n_pool] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n_pool + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if not res.success:
        w = np.full(n_pool, 1.0 / n_pool)
        per_sta = w @ R
        return w, float(per_sta.min()), per_sta

    w = np.maximum(res.x[:n_pool], 0.0)
    s = w.sum()
    if s > eps_feas:
        w = w / s
    t_opt   = float(-res.fun)
    per_sta = w @ R
    return w, max(t_opt, 0.0), per_sta


# ---------------------------------------------------------------------------
# Per-station evaluator (JIT) — one config -> rate vector (Mb/s)
# ---------------------------------------------------------------------------

def make_per_sta_evaluator(scenario, info: ScenarioInfo):
    to_arrays  = make_config_to_arrays(info)
    ap_ids     = jnp.array(info.ap_ids, dtype=jnp.int32)
    max_stas   = info.max_stas
    valid_flat = jnp.array(np.flatnonzero(info.valid_mask), dtype=jnp.int32)

    @jax.jit
    def _eval(config: NetworkConfig, key: jax.Array) -> jax.Array:
        tx, tx_p, mcs = to_arrays(config)
        result    = scenario(key, tx, tx_p, mcs, return_internals=True)
        internals = result[2]
        ap_rate   = internals.average_data_rate[ap_ids] / jnp.float32(1e6)   # (n_aps,)
        ap_rate_x = jnp.repeat(ap_rate, max_stas)                             # (n_aps*max_stas,)
        sel_flat  = config.selected.ravel().astype(jnp.float32)
        rate_flat = sel_flat * ap_rate_x
        return rate_flat[valid_flat]   # (n_stas,)

    return _eval


# ---------------------------------------------------------------------------
# Weighted-objective SA pricer
# ---------------------------------------------------------------------------

class _PSAState(NamedTuple):
    config:      NetworkConfig
    cur_obj:     jax.Array
    cur_per_sta: jax.Array
    best_config: NetworkConfig
    best_obj:    jax.Array
    best_per_sta: jax.Array
    key:         jax.Array


def make_pricer(
    scenario,
    info:    ScenarioInfo,
    n_steps: int,
    T_0:     float,
    T_decay: float,
):
    """Return fn (lambda_vec: np.ndarray, init_config: NetworkConfig, key)
    -> (best_config, best_per_sta, best_obj).

    Maximises dot(lambda, per_sta_rates). Reuses the throughput SA neighbour
    operator; only the scalar objective changes.
    """
    eval_per_sta = make_per_sta_evaluator(scenario, info)
    valid_mask   = jnp.array(info.valid_mask, dtype=jnp.int32)
    T_0_f        = jnp.float32(T_0)
    T_decay_f    = jnp.float32(T_decay)

    def _step(state: _PSAState, carry):
        step_idx, lam = carry
        T = T_0_f * (T_decay_f ** step_idx.astype(jnp.float32))
        key, nbr_key, sim_key, acc_key = jax.random.split(state.key, 4)

        cand       = neighbor(state.config, nbr_key, valid_mask)
        cand_rates = eval_per_sta(cand, sim_key)
        cand_obj   = jnp.sum(lam * cand_rates)

        delta      = cand_obj - state.cur_obj
        log_accept = jnp.minimum(jnp.float32(0.0), delta / jnp.maximum(T, jnp.float32(1e-30)))
        accept     = jnp.log(jax.random.uniform(acc_key)) < log_accept

        new_cfg   = jax.lax.cond(accept, lambda: cand, lambda: state.config)
        new_obj   = jnp.where(accept, cand_obj,   state.cur_obj)
        new_rates = jax.lax.cond(accept, lambda: cand_rates, lambda: state.cur_per_sta)

        improved = new_obj > state.best_obj
        bcfg, bobj, brates = jax.lax.cond(
            improved,
            lambda: (new_cfg, new_obj, new_rates),
            lambda: (state.best_config, state.best_obj, state.best_per_sta),
        )
        return _PSAState(new_cfg, new_obj, new_rates, bcfg, bobj, brates, key), None

    @jax.jit
    def _run(lam: jax.Array, init_config: NetworkConfig, key: jax.Array):
        key, eval_key = jax.random.split(key)
        init_rates = eval_per_sta(init_config, eval_key)
        init_obj   = jnp.sum(lam * init_rates)
        init = _PSAState(
            config=init_config, cur_obj=init_obj, cur_per_sta=init_rates,
            best_config=init_config, best_obj=init_obj, best_per_sta=init_rates,
            key=key,
        )
        steps  = jnp.arange(n_steps, dtype=jnp.int32)
        lam_b  = jnp.broadcast_to(lam, (n_steps, lam.shape[0]))
        final, _ = jax.lax.scan(_step, init, (steps, lam_b))
        return final.best_config, final.best_per_sta, final.best_obj

    def pricer(lam_np: np.ndarray, init_config: NetworkConfig, key: jax.Array):
        lam = jnp.asarray(lam_np, dtype=jnp.float32)
        return _run(lam, init_config, key)

    return pricer


# ---------------------------------------------------------------------------
# Pool initialisation — one basis config per station guarantees coverage
# ---------------------------------------------------------------------------

def _make_basis_pool(info: ScenarioInfo, key: jax.Array) -> list[NetworkConfig]:
    """Single-AP-active basis: one config per station with ONLY the serving AP
    active, all others silent. Interference-free, so each station's rate in its
    own basis column is guaranteed > 0 at max power / reasonable MCS.

    This matters for bootstrapping: if the basis gives every station a positive
    rate, the first LP has t > 0 and the bottleneck-uniform lambda is well-posed.
    """
    valid_pairs = np.argwhere(info.valid_mask)       # (n_stas, 2)
    n_aps, max_stas = info.n_aps, info.max_stas
    keys = jax.random.split(key, len(valid_pairs))

    pool: list[NetworkConfig] = []
    for i, (ap_idx, sta_slot) in enumerate(valid_pairs):
        ap_idx, sta_slot = int(ap_idx), int(sta_slot)
        # selected: only (ap_idx, sta_slot) = 1
        selected = jnp.zeros((n_aps, max_stas), dtype=jnp.int32).at[ap_idx, sta_slot].set(1)
        # tx_power: max (3); mcs: medium-high (9) as a safe-high default.
        tx_power = jnp.full((n_aps, max_stas), 3, dtype=jnp.int32)
        mcs      = jnp.full((n_aps, max_stas), 9, dtype=jnp.int32)
        pool.append(NetworkConfig(selected, tx_power, mcs))
    return pool


# ---------------------------------------------------------------------------
# Lambda strategies (primal, no duals)
# ---------------------------------------------------------------------------

def _lambda_bottleneck_uniform(
    per_sta: np.ndarray, t: float, eps_bot: float,
) -> np.ndarray:
    """Uniform over the bottleneck set B = {s : r_s <= t + eps}."""
    B = per_sta <= (t + eps_bot)
    if not B.any():
        B[np.argmin(per_sta)] = True
    lam = B.astype(np.float32)
    return lam / lam.sum()


def _lambda_inverse_gap(
    per_sta: np.ndarray, t: float, eps_gap: float,
) -> np.ndarray:
    """Smooth: lam_s ∝ 1 / (r_s - t + eps). Concentrates on tight stations."""
    gap = np.maximum(per_sta - t, 0.0) + eps_gap
    lam = 1.0 / gap
    return (lam / lam.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    scenario,
    *,
    seed:           int   = 42,
    n_outer:        int   = 50,
    n_steps:        int   = 1000,
    patience:       int   = 5,
    max_pool:       int   = 64,
    T_0:            float = 5.0,
    T_decay:        float = 0.999,
    eps_bot:        float = 1e-3,     # Mb/s tolerance for bottleneck set
    lambda_mode:    str   = 'bottleneck',   # 'bottleneck' | 'inverse_gap'
    pricing_diversify_on_stall: bool = True,
    top_n:          int   = 10,        # ignored, accepted for interface parity
) -> FResult:
    """Run primal-only column generation for max-min fairness."""
    info         = make_scenario_info(scenario)
    eval_per_sta = make_per_sta_evaluator(scenario, info)
    pricer       = make_pricer(scenario, info, n_steps=n_steps, T_0=T_0, T_decay=T_decay)
    rng_np       = np.random.default_rng(seed)

    master_key = jax.random.PRNGKey(seed)
    master_key, basis_key, eval_key = jax.random.split(master_key, 3)

    # --- seed the pool with basis configs, evaluate them ---
    pool_cfgs:  list[NetworkConfig] = _make_basis_pool(info, basis_key)
    R_rows:     list[np.ndarray]    = []
    for c in pool_cfgs:
        eval_key, k = jax.random.split(eval_key)
        R_rows.append(np.asarray(eval_per_sta(c, k)))
    R = np.stack(R_rows, axis=0)   # (|P|, n_stas)

    # --- outer CG loop ---
    best_min_history: list[float] = []
    best_sum_history: list[float] = []
    score_history:    list[float] = []

    w, t, per_sta = solve_max_min_lp(R)
    best_t        = t
    best_w        = w.copy()
    best_per_sta  = per_sta.copy()
    best_R        = R.copy()
    best_pool     = list(pool_cfgs)
    stalls        = 0
    stall_kick    = 0

    for it in range(n_outer):
        # --- choose lambda (no duals) ---
        if lambda_mode == 'inverse_gap':
            lam = _lambda_inverse_gap(per_sta, t, eps_gap=1.0)
        else:
            lam = _lambda_bottleneck_uniform(per_sta, t, eps_bot)

        # Diversification on consecutive stalls: randomly reweight lambda within B.
        if pricing_diversify_on_stall and stall_kick > 0:
            noise = rng_np.random(lam.shape).astype(np.float32) * (0.5 * stall_kick)
            lam   = lam * (1.0 + noise)
            # Keep lam on the same support sign pattern; renormalise.
            if lam.sum() > 0: lam = lam / lam.sum()

        # --- pricing: warm start from a random pool member ---
        warm_idx = int(rng_np.integers(0, len(pool_cfgs)))
        master_key, p_key = jax.random.split(master_key)
        c_new, r_new, _obj = pricer(lam, pool_cfgs[warm_idx], p_key)
        r_new_np = np.asarray(r_new)

        # --- primal admission test: does the RMP improve? ---
        R_trial = np.vstack([R, r_new_np[None, :]])
        w_t, t_t, per_sta_t = solve_max_min_lp(R_trial)

        improved = t_t > best_t + 1e-6
        if improved:
            pool_cfgs.append(c_new)
            R       = R_trial
            w, t, per_sta = w_t, t_t, per_sta_t
            best_t, best_w, best_per_sta = t, w.copy(), per_sta.copy()
            best_R, best_pool = R.copy(), list(pool_cfgs)
            stalls, stall_kick = 0, 0
        else:
            stalls     += 1
            stall_kick += 1
            if stalls >= patience:
                # Record final histories and stop early.
                best_min_history.append(float(best_per_sta.min()))
                best_sum_history.append(float(best_per_sta.sum()))
                score_history.append(float(best_per_sta.min()) * 1e3 + float(best_per_sta.sum()))
                break

        # --- pool size cap: drop columns with ~zero weight in current RMP ---
        if len(pool_cfgs) > max_pool:
            keep = w > 1e-8
            # always keep at least n_stas columns (feasibility guard)
            if keep.sum() >= R.shape[1]:
                pool_cfgs = [pool_cfgs[i] for i in range(len(pool_cfgs)) if keep[i]]
                R = R[keep]
                w, t, per_sta = solve_max_min_lp(R)
                best_t, best_w, best_per_sta = t, w.copy(), per_sta.copy()
                best_R, best_pool = R.copy(), list(pool_cfgs)

        # --- history bookkeeping ---
        best_min_history.append(float(best_per_sta.min()))
        best_sum_history.append(float(best_per_sta.sum()))
        score_history.append(float(best_per_sta.min()) * 1e3 + float(best_per_sta.sum()))

    # --- final sanity on best solution: all stations covered, rates >= 0 ---
    # The basis pool guarantees each station is served by at least one column,
    # and LP gives a feasible w >= 0.  A degenerate LP still returns a valid
    # per_sta, so no extra coverage check is needed here.

    best_per_sta_np = np.asarray(best_per_sta)

    # Packed "best_solution" field (for interface parity): return the best pool
    # laid out into an FSolution-like tuple.  Downstream plotting only needs
    # the history / scalars, so we just return None and let the caller ignore.
    return FResult(
        best_solution = None,                                 # noqa: typing
        best_score    = float(best_per_sta_np.min()) * 1e3 + float(best_per_sta_np.sum()),
        best_min_rate = float(best_per_sta_np.min()),
        best_sum_rate = float(best_per_sta_np.sum()),
        best_fairness = jains_index(best_per_sta_np),
        best_per_sta  = best_per_sta_np.tolist(),
        history       = score_history,
        min_history   = best_min_history,
        sum_history   = best_sum_history,
        info          = info,
    )
