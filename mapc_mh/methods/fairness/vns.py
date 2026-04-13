"""Variable Neighbourhood Search for F-Optimal (fairness) approximation.

Algorithm: Basic VNS (BVNS) with two-scale neighbourhood structure
--------------------------------------------------------------------
The outer loop cycles through k=1..k_max structural shake neighbourhoods.
After each shake, a fine-grained hill-climbing local search is applied.
If the result improves the current solution, accept and reset k=1;
otherwise increment k (mod k_max).

Two-scale design
----------------
  Local search — fine-grained tuning, one thing at a time (1/3 each):
    * mutate_mcs         — change MCS of one active AP in a randomly chosen live sub-config
    * mutate_tx_power    — change tx_power of one active AP in a randomly chosen live sub-config
    * perturb_one_logit  — add Gaussian noise to a single randomly chosen logit
    No coverage check needed (assignments never touched).

  Shake neighbourhoods — structural changes that escape local optima:
    k=1  mutate_selected  — change AP-STA assignment in one live sub-config
    k=2  remove_config    — deactivate a live slot (no-op if only one active)
    k=3  add_config       — revive the weakest slot with a fresh random config

JAX compilation strategy
------------------------
Both the outer loop and the local search are compiled via jax.lax.scan (same as SA).
The neighbourhood index k is a traced JAX int32 dispatched via jax.lax.switch.
The local search uses a 'stuck' flag to emulate early-break without leaving JIT.
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from mapc_mh.config import NetworkConfig, ScenarioInfo, make_random_config
from mapc_mh.methods.fairness.core import (
    FSolution, FResult,
    LIVE_EPS, PERTURB_SIGMA, REVIVE_LOGIT_DELTA, REMOVE_LOGIT, REJECTION_CAP,
    jains_index, leximin_delta, leximin_score,
    make_coverage_check, setup_f,
)
from mapc_mh.methods.fairness.weights import WeightStrategy, identity_strategy


# ---------------------------------------------------------------------------
# Fine-grained neighbour for local search (mcs + tx_power only)
# ---------------------------------------------------------------------------

def make_fine_neighbor_f(
    info:        ScenarioInfo,
    max_configs: int,
) -> Callable[[FSolution, jax.Array], FSolution]:
    """Return a fine-grained neighbour generator for use in local search.

    Randomly applies one of three moves with equal probability (1/3 each):
      0  mutate_mcs        — change MCS of one active AP in one live sub-config
      1  mutate_tx_power   — change tx_power of one active AP in one live sub-config
      2  perturb_one_logit — add Gaussian noise to a single randomly chosen logit

    No coverage rejection needed — assignments (selected) are never touched.
    """
    n_aps, max_stas = info.valid_mask.shape

    def _fn(sol: FSolution, key: jax.Array) -> FSolution:
        weights = jax.nn.softmax(sol.logits)
        key, slot_key, op_key, ap_key, val_key, logit_key = jax.random.split(key, 6)
        chosen = jax.random.choice(slot_key, max_configs, p=weights)

        ap_is_active     = jnp.any(sol.configs.selected[chosen] > 0, axis=-1)
        selected_sta_idx = jnp.argmax(sol.configs.selected[chosen], axis=-1)
        active_probs     = ap_is_active.astype(jnp.float32)
        active_probs     = active_probs / (active_probs.sum() + jnp.float32(1e-9))
        ap_idx           = jax.random.choice(ap_key, n_aps, p=active_probs)
        sta_idx          = selected_sta_idx[ap_idx]

        def mutate_mcs(_):
            new_val = jax.random.randint(val_key, (), 0, 14)
            new_mcs = sol.configs.mcs.at[chosen, ap_idx, sta_idx].set(new_val)
            return FSolution(configs=NetworkConfig(sol.configs.selected, sol.configs.tx_power, new_mcs),
                             logits=sol.logits)

        def mutate_tx_power(_):
            new_val = jax.random.randint(val_key, (), 0, 4)
            new_tp  = sol.configs.tx_power.at[chosen, ap_idx, sta_idx].set(new_val)
            return FSolution(configs=NetworkConfig(sol.configs.selected, new_tp, sol.configs.mcs),
                             logits=sol.logits)

        def perturb_one_logit(_):
            logit_idx  = jax.random.randint(logit_key, (), 0, max_configs)
            noise      = jax.random.normal(val_key, ()) * jnp.float32(PERTURB_SIGMA)
            new_logits = sol.logits.at[logit_idx].set(sol.logits[logit_idx] + noise)
            return FSolution(configs=sol.configs, logits=new_logits)

        op = jax.random.randint(op_key, (), 0, 3)
        return jax.lax.switch(op, [mutate_mcs, mutate_tx_power, perturb_one_logit], None)

    return _fn


# ---------------------------------------------------------------------------
# Structural shake functions (no @jax.jit — used inside jax.lax.switch)
# ---------------------------------------------------------------------------

def make_shakers(
    info:        ScenarioInfo,
    max_configs: int,
) -> list[Callable]:
    """Return [shake_k1, shake_k2, shake_k3, shake_k4] for use with jax.lax.switch.

    Each function: (sol: FSolution, key: jax.Array) -> FSolution.
    Shakes k=1..3 embed a coverage rejection loop (up to REJECTION_CAP retries);
    if exhausted, returns sol unchanged (no-op — k increments in outer loop).
    shake_k4 (add_config) always produces a valid solution by construction.
    Note: no @jax.jit — called inside an outer jax.lax.scan.

    k=1  mutate_selected  — change AP-STA assignment in one live sub-config
    k=2  remove_config    — deactivate a live slot (no-op if only one active)
    k=3  add_config       — revive weakest slot with a fresh random config
    """
    valid_mask       = jnp.array(info.valid_mask, dtype=jnp.int32)
    n_aps, max_stas  = valid_mask.shape
    all_covered      = make_coverage_check(info)
    random_config_fn = make_random_config(info)

    def _with_coverage_retry(sample_fn, sol, key):
        key, init_key  = jax.random.split(key)
        init_candidate = sample_fn(sol, init_key)

        def _cond(state):
            candidate, _, tries = state
            return (~all_covered(candidate)) & (tries < REJECTION_CAP)

        def _body(state):
            _, key, tries = state
            key, sub_key = jax.random.split(key)
            return sample_fn(sol, sub_key), key, tries + 1

        candidate, _, _ = jax.lax.while_loop(
            _cond, _body, (init_candidate, key, jnp.int32(1))
        )
        return jax.lax.cond(all_covered(candidate), lambda: candidate, lambda: sol)

    # --- k=1: mutate_selected in one live sub-config ---
    def shake_k1(sol: FSolution, key: jax.Array) -> FSolution:
        def _sample(sol, key):
            weights = jax.nn.softmax(sol.logits)
            key, slot_key, ap_key, action_key, val_key = jax.random.split(key, 5)
            chosen = jax.random.choice(slot_key, max_configs, p=weights)

            config           = NetworkConfig(sol.configs.selected[chosen],
                                             sol.configs.tx_power[chosen],
                                             sol.configs.mcs[chosen])
            ap_is_active     = jnp.any(config.selected > 0, axis=-1)
            n_active         = jnp.sum(ap_is_active)
            selected_sta_idx = jnp.argmax(config.selected, axis=-1)
            n_valid_per_ap   = jnp.sum(valid_mask, axis=-1)
            ap_idx           = jax.random.randint(ap_key, (), 0, n_aps)
            is_active        = ap_is_active[ap_idx]
            cur_sta          = selected_sta_idx[ap_idx]
            n_valid          = n_valid_per_ap[ap_idx]

            def _activate(_):
                probs   = valid_mask[ap_idx].astype(jnp.float32)
                probs   = probs / (probs.sum() + jnp.float32(1e-9))
                new_sta = jax.random.choice(val_key, max_stas, p=probs)
                new_row = jnp.zeros(max_stas, dtype=jnp.int32).at[new_sta].set(1)
                new_sel = sol.configs.selected.at[chosen, ap_idx].set(new_row)
                return FSolution(configs=NetworkConfig(new_sel, sol.configs.tx_power, sol.configs.mcs),
                                 logits=sol.logits)

            def _deactivate(_):
                new_row = jnp.zeros(max_stas, dtype=jnp.int32)
                new_sel = jax.lax.cond(
                    n_active > 1,
                    lambda: sol.configs.selected.at[chosen, ap_idx].set(new_row),
                    lambda: sol.configs.selected,
                )
                return FSolution(configs=NetworkConfig(new_sel, sol.configs.tx_power, sol.configs.mcs),
                                 logits=sol.logits)

            def _switch(_):
                probs   = valid_mask[ap_idx].astype(jnp.float32).at[cur_sta].set(0.0)
                probs   = probs / (probs.sum() + jnp.float32(1e-9))
                new_sta = jax.random.choice(val_key, max_stas, p=probs)
                new_row = jnp.zeros(max_stas, dtype=jnp.int32).at[new_sta].set(1)
                new_sel = jax.lax.cond(
                    n_valid > 1,
                    lambda: sol.configs.selected.at[chosen, ap_idx].set(new_row),
                    lambda: sol.configs.selected,
                )
                return FSolution(configs=NetworkConfig(new_sel, sol.configs.tx_power, sol.configs.mcs),
                                 logits=sol.logits)

            def when_active(_):
                do_switch = jax.random.randint(action_key, (), 0, 2)
                return jax.lax.cond(do_switch == 1, _switch, _deactivate, None)

            return jax.lax.cond(is_active, when_active, _activate, None)

        return _with_coverage_retry(_sample, sol, key)

    # --- k=2: remove_config ---
    def shake_k2(sol: FSolution, key: jax.Array) -> FSolution:
        def _sample(sol, key):
            weights    = jax.nn.softmax(sol.logits)
            n_live     = jnp.sum(weights > LIVE_EPS).astype(jnp.float32)
            key, slot_key = jax.random.split(key)
            chosen     = jax.random.choice(slot_key, max_configs, p=weights)
            new_logits = sol.logits.at[chosen].set(jnp.float32(REMOVE_LOGIT))
            removed    = FSolution(configs=sol.configs, logits=new_logits)
            return jax.lax.cond(n_live > jnp.float32(1.0), lambda: removed, lambda: sol)
        return _with_coverage_retry(_sample, sol, key)

    # --- k=3: add_config ---
    def shake_k3(sol: FSolution, key: jax.Array) -> FSolution:
        # add_config always preserves coverage — no rejection loop needed
        weights    = jax.nn.softmax(sol.logits)
        key, cfg_key = jax.random.split(key)
        weakest    = jnp.argmin(weights)
        new_cfg    = random_config_fn(cfg_key)
        new_sel    = sol.configs.selected.at[weakest].set(new_cfg.selected)
        new_tp     = sol.configs.tx_power.at[weakest].set(new_cfg.tx_power)
        new_mcs    = sol.configs.mcs.at[weakest].set(new_cfg.mcs)
        new_logit  = jnp.max(sol.logits) + jnp.float32(REVIVE_LOGIT_DELTA)
        new_logits = sol.logits.at[weakest].set(new_logit)
        return FSolution(configs=NetworkConfig(new_sel, new_tp, new_mcs), logits=new_logits)

    return [shake_k1, shake_k2, shake_k3]


# ---------------------------------------------------------------------------
# JIT-compiled runner (lax.scan over outer VNS steps)
# ---------------------------------------------------------------------------

def _make_runner(evaluate, fine_neighbor_f, shakers, k_max, local_search_steps, n_steps):
    """Return a JIT-compiled VNS runner.

    The outer loop uses jax.lax.scan; k is a traced int32 dispatched via
    jax.lax.switch. The local search uses jax.lax.scan with a 'stuck' flag to
    emulate early-break without leaving the JIT context.
    """

    def _ls_step(carry, _):
        """One step of fine-grained hill-climbing local search."""
        sol, per_sta, key, stuck = carry
        key, nbr_key, eval_key = jax.random.split(key, 3)
        candidate   = fine_neighbor_f(sol, nbr_key)
        c_per_sta, _, _ = evaluate(candidate, eval_key)
        improved    = leximin_delta(c_per_sta, per_sta) > jnp.float32(0.0)
        do_update   = improved & ~stuck
        new_sol     = jax.lax.cond(do_update, lambda: candidate, lambda: sol)
        new_per_sta = jax.lax.cond(do_update, lambda: c_per_sta, lambda: per_sta)
        new_stuck   = stuck | ~improved
        return (new_sol, new_per_sta, key, new_stuck), None

    def _vns_step(state, _):
        """One outer VNS iteration: shake → local search → acceptance → best update."""
        sol, per_sta, best_sol, best_per_sta, best_score, k, key = state
        key, shake_key, eval_key, ls_key = jax.random.split(key, 4)

        # Shake: force a structural perturbation using k-th neighbourhood
        x_prime = jax.lax.switch(k - 1, shakers, sol, shake_key)
        x_prime_per_sta, _, _ = evaluate(x_prime, eval_key)

        # Local search: fine-grained hill climbing from shaken solution
        (x_dp, x_dp_per_sta, ls_key, _), _ = jax.lax.scan(
            _ls_step,
            (x_prime, x_prime_per_sta, ls_key, jnp.bool_(False)),
            None,
            length=local_search_steps,
        )

        # VNS acceptance
        vns_better  = leximin_delta(x_dp_per_sta, per_sta) > jnp.float32(0.0)
        new_sol     = jax.lax.cond(vns_better, lambda: x_dp,         lambda: sol)
        new_per_sta = jax.lax.cond(vns_better, lambda: x_dp_per_sta, lambda: per_sta)
        new_k       = jax.lax.cond(
            vns_better,
            lambda: jnp.int32(1),
            lambda: k % jnp.int32(k_max) + jnp.int32(1),
        )

        # Update global best
        best_better      = leximin_delta(new_per_sta, best_per_sta) > jnp.float32(0.0)
        new_best_sol     = jax.lax.cond(best_better, lambda: new_sol,      lambda: best_sol)
        new_best_per_sta = jax.lax.cond(best_better, lambda: new_per_sta,  lambda: best_per_sta)
        new_best_score   = jax.lax.cond(
            best_better,
            lambda: leximin_score(new_best_per_sta),
            lambda: best_score,
        )

        new_state = (new_sol, new_per_sta, new_best_sol, new_best_per_sta, new_best_score, new_k, key)
        outputs   = (jnp.min(new_best_per_sta), jnp.sum(new_best_per_sta), new_best_score)
        return new_state, outputs

    @jax.jit
    def _run(key, initial_sol, initial_per_sta):
        init_score = leximin_score(initial_per_sta)
        init_state = (
            initial_sol, initial_per_sta,
            initial_sol, initial_per_sta, init_score,
            jnp.int32(1), key,
        )
        final_state, (min_hist, sum_hist, score_hist) = jax.lax.scan(
            _vns_step, init_state, None, length=n_steps
        )
        _, _, best_sol, best_per_sta, best_score, _, _ = final_state
        return best_sol, best_per_sta, best_score, min_hist, sum_hist, score_hist

    def runner(key, initial_sol, initial_per_sta):
        best_sol, best_per_sta, best_score, min_hist, sum_hist, score_hist = _run(
            key, initial_sol, initial_per_sta
        )
        return (
            best_sol,
            best_per_sta,
            float(best_score),
            min_hist.tolist(),
            sum_hist.tolist(),
            score_hist.tolist(),
        )

    return runner


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    scenario,
    *,
    seed:               int            = 42,
    n_steps:            int            = 2000,
    top_n:              int            = 10,       # interface parity with F-SA, unused
    k_max:              int            = 3,
    local_search_steps: int            = 20,
    max_configs:        int            = 32,
    weight_strategy:    WeightStrategy = None,
) -> FResult:
    """Run Variable Neighbourhood Search for F-Optimal (fairness) approximation.

    Parameters
    ----------
    scenario           : simulator callable (StaticScenario)
    seed               : RNG seed
    n_steps            : number of outer VNS iterations (JIT-compiled via lax.scan)
    k_max              : number of shake neighbourhoods (1..k_max, max=3)
    local_search_steps : fine-grained LS steps per shake (mcs/tx_power only)
    max_configs        : compile-time slot count (upper bound on sub-configs)
    weight_strategy    : WeightStrategy callable; defaults to identity_strategy
    top_n              : accepted for interface parity, unused by F-VNS
    """
    if weight_strategy is None:
        weight_strategy = identity_strategy

    info, evaluate, run_key, initial_sol, initial_per_sta, max_configs = setup_f(
        scenario, seed, max_configs, weight_strategy
    )

    fine_neighbor_f = make_fine_neighbor_f(info, max_configs)
    shakers         = make_shakers(info, max_configs)

    runner = _make_runner(evaluate, fine_neighbor_f, shakers, k_max, local_search_steps, n_steps)

    best_sol, best_per_sta, best_score, min_hist, sum_hist, score_hist = runner(
        run_key, initial_sol, initial_per_sta
    )

    best_per_sta_np = np.array(best_per_sta)

    check = make_coverage_check(info)
    assert bool(check(best_sol)), (
        "F-VNS returned a best solution that does not cover all stations."
    )

    return FResult(
        best_solution = best_sol,
        best_score    = best_score,
        best_min_rate = float(best_per_sta_np.min()),
        best_sum_rate = float(best_per_sta_np.sum()),
        best_fairness = jains_index(best_per_sta_np),
        best_per_sta  = best_per_sta_np.tolist(),
        history       = score_hist,
        min_history   = min_hist,
        sum_history   = sum_hist,
        info          = info,
    )
