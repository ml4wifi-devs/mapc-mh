"""Core primitives for the F-Optimal (fairness) metaheuristic family.

Solution representation
-----------------------
An FSolution holds a fixed-size batch of Co-SR sub-configurations (NetworkConfig)
and a logit vector. Weights are always softmax(logits), which lies on the simplex.
Unused slots are driven toward very negative logits by the search; there is no hard
cap on the "active" count — the optimizer naturally regularises it via zero weights.

Evaluation
----------
All sub-configs are evaluated in parallel via jax.vmap over the fixed leading
dimension. Inactive slots (near-zero weight) contribute negligibly to the per-station
weighted mean. No threads or processes are used anywhere.

Objective: lexicographic max-min
---------------------------------
The acceptance criterion in SA (and other trajectory algorithms) uses leximin_delta,
which sorts per-station throughput vectors ascending and finds the first index where
they differ. Identical vectors yield delta=0, which SA accepts (exploration).

Neighbourhood operators
-----------------------
Four moves (modify_config, perturb_weights, add_config, remove_config) dispatched via
jax.lax.switch. Move probabilities are proportional to approximate neighbourhood size,
so that parameter-level modifications dominate structural changes as the solution grows.

LP ablation seam
----------------
The evaluator accepts a WeightStrategy callable (see weights.py). The default
identity_strategy lets the metaheuristic control logits directly. Swap in an LP
strategy later to freeze config structure and optimise weights analytically.
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mapc_mh.config import (
    NetworkConfig, ScenarioInfo,
    make_config_to_arrays, make_random_config, make_scenario_info,
)
from mapc_mh.methods.core import neighbor


# ---------------------------------------------------------------------------
# Module-level knobs (import and tweak in VNS/GA/Memetic as needed)
# ---------------------------------------------------------------------------

LIVE_EPS           = 1e-4   # weight threshold below which a slot is considered inactive
PERTURB_SIGMA      = 0.5    # std of Gaussian noise added to logits in perturb_weights
REVIVE_LOGIT_DELTA = -0.693 # ln(0.5): new slot gets ~half the weight of the strongest
REMOVE_LOGIT       = -30.0  # logit value that effectively zeros a slot's weight
REJECTION_CAP      = 32     # maximum retries in the coverage rejection loop


# ---------------------------------------------------------------------------
# Solution type
# ---------------------------------------------------------------------------

class FSolution(NamedTuple):
    """A weighted set of Co-SR sub-configurations for F-Optimal search.

    configs : NetworkConfig — batched over a fixed leading axis of size max_configs.
              Each field (selected, tx_power, mcs) has shape (max_configs, n_aps, max_stas).
    logits  : (max_configs,) float32 — pre-softmax weights.
              weights = softmax(logits) ∈ simplex.
              Inactive slots have logits ≈ -30 → weight ≈ 0.
    """
    configs: NetworkConfig
    logits:  jax.Array


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FResult:
    best_solution:   FSolution
    best_score:      float        # leximin surrogate: min_rate * 1e3 + sum_rate
    best_min_rate:   float        # raw max-min throughput (Mb/s)
    best_sum_rate:   float        # total throughput (Mb/s), for comparison vs T-Optimal
    best_fairness:   float        # Jain's fairness index over per-station rates
    best_per_sta:    list[float]  # per-station rates at the best solution
    history:         list[float]  # surrogate score per step
    min_history:     list[float]  # min per-station throughput per step (best-so-far)
    sum_history:     list[float]  # total throughput per step (best-so-far)
    info:            ScenarioInfo | None = None


def jains_index(per_sta: np.ndarray) -> float:
    """Jain's fairness index: (sum x_i)^2 / (n * sum x_i^2). Returns 1.0 for empty/zero input."""
    s  = per_sta.sum()
    s2 = (per_sta ** 2).sum()
    return float((s * s) / (len(per_sta) * s2)) if s2 > 0 else 1.0


# ---------------------------------------------------------------------------
# Coverage predicate (JIT-friendly)
# ---------------------------------------------------------------------------

def make_coverage_check(info: ScenarioInfo) -> Callable[[FSolution], jax.Array]:
    """Return a JIT-compiled fn: FSolution -> bool  (True = all stations covered)."""
    valid_mask = jnp.array(info.valid_mask, dtype=jnp.bool_)  # (n_aps, max_stas)

    @jax.jit
    def _all_covered(sol: FSolution) -> jax.Array:
        weights  = jax.nn.softmax(sol.logits)                        # (M,)
        live     = (weights > LIVE_EPS)[:, None, None]               # (M, 1, 1)
        # selected: (M, n_aps, max_stas) — 1 if that sub-config assigns AP to STA
        contrib  = (sol.configs.selected * live).sum(axis=0) > 0     # (n_aps, max_stas)
        return jnp.all(contrib | ~valid_mask)

    return _all_covered


# ---------------------------------------------------------------------------
# Initial solution
# ---------------------------------------------------------------------------

def make_random_f_solution(
    info: ScenarioInfo,
    max_configs: int,
    key: jax.Array,
) -> FSolution:
    """Generate an FSolution guaranteed to cover all stations.

    Strategy: for each valid (ap_idx, sta_slot) pair, create one "basis" sub-config
    that forces exactly that station to be active (the AP is active and assigned to
    that STA slot; other APs are randomised). This guarantees full station coverage
    by construction. Remaining slots (max_configs - n_stas) are filled with fresh
    random configs at logit=-30 (inactive initially).

    Active slots (basis configs) get logit=0 so softmax gives uniform weights.
    The search then explores weight perturbations and config modifications.
    """
    random_config_fn = make_random_config(info)
    n_aps, max_stas = info.n_aps, info.max_stas

    # Enumerate valid (ap_idx, sta_slot) pairs — these are the n_stas stations
    valid_pairs = np.argwhere(info.valid_mask)           # (n_stas, 2)  numpy
    n_stas      = len(valid_pairs)

    if n_stas > max_configs:
        raise ValueError(
            f'max_configs={max_configs} < n_stas={n_stas}. '
            f'Increase max_configs so every station can have a basis config.'
        )

    @jax.jit
    def _make(key: jax.Array) -> FSolution:
        key, rand_key = jax.random.split(key)

        # --- Basis configs: one per station, guaranteed to cover it ---
        basis_keys = jax.random.split(key, n_stas)

        def make_basis(ap_idx_sta_slot, bkey):
            ap_idx, sta_slot = ap_idx_sta_slot[0], ap_idx_sta_slot[1]
            base  = random_config_fn(bkey)  # random config as starting point
            # Force-assign: AP ap_idx serves sta_slot
            forced_row = (
                jnp.zeros(max_stas, dtype=jnp.int32).at[sta_slot].set(1)
            )
            new_selected = base.selected.at[ap_idx].set(forced_row)
            return NetworkConfig(new_selected, base.tx_power, base.mcs)

        pairs_jnp = jnp.array(valid_pairs, dtype=jnp.int32)       # (n_stas, 2)
        basis_configs = jax.vmap(make_basis)(pairs_jnp, basis_keys)  # batched

        # --- Filler configs: random, will be inactive at logit=-30 ---
        n_filler  = max_configs - n_stas
        fill_keys = jax.random.split(rand_key, max(n_filler, 1))
        filler_configs = jax.vmap(random_config_fn)(fill_keys[:max(n_filler, 1)])

        # --- Stack into fixed-size batch ---
        if n_filler > 0:
            all_selected = jnp.concatenate([basis_configs.selected, filler_configs.selected], axis=0)
            all_tp       = jnp.concatenate([basis_configs.tx_power, filler_configs.tx_power], axis=0)
            all_mcs      = jnp.concatenate([basis_configs.mcs,      filler_configs.mcs],      axis=0)
        else:
            all_selected = basis_configs.selected
            all_tp       = basis_configs.tx_power
            all_mcs      = basis_configs.mcs

        configs = NetworkConfig(all_selected, all_tp, all_mcs)

        # Basis slots get logit=0 (active, uniform weight); fillers get logit=-30
        logits = jnp.concatenate([
            jnp.zeros(n_stas, dtype=jnp.float32),
            jnp.full(max(n_filler, 0), REMOVE_LOGIT, dtype=jnp.float32),
        ])

        return FSolution(configs=configs, logits=logits)

    return _make(key)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def make_evaluator(
    scenario,
    info: ScenarioInfo,
    max_configs: int,
    weight_strategy,   # WeightStrategy from weights.py
) -> Callable[[FSolution, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """Return a JIT-compiled evaluator: (FSolution, key) -> (per_sta, sub_sta_rates, weights).

    per_sta       : (n_stas,) — weighted mean per-station throughput in Mb/s
    sub_sta_rates : (max_configs, n_stas) — per sub-config per-station throughput
    weights       : (max_configs,) — softmax weights actually used

    Per-station throughput extraction
    ----------------------------------
    The simulator's `internals.average_data_rate` is indexed by TRANSMITTER node and
    reported in bits/s. For APs, `average_data_rate[ap_node_id]` equals the throughput
    received by the AP's currently assigned STA. We extract per-station rates as:

        rate_at(ap_idx, sta_slot) = selected[ap_idx, sta_slot]
                                    * average_data_rate[ap_ids[ap_idx]] / 1e6   [Mb/s]

    Only the assigned (selected=1) slot of each AP contributes; all other slots are 0.
    Stations not assigned in a sub-config receive 0 from that sub-config's rate.
    The final per-station vector is the weighted mean across all sub-configs.
    """
    to_arrays  = make_config_to_arrays(info)
    ap_ids_jnp = jnp.array(info.ap_ids, dtype=jnp.int32)   # (n_aps,)
    n_aps      = info.n_aps
    max_stas   = info.max_stas

    # Flat indices into the (n_aps * max_stas,) array for valid (ap, sta) slots
    valid_flat_idx = jnp.array(
        np.flatnonzero(info.valid_mask), dtype=jnp.int32
    )  # (n_stas,)

    def _eval_one(config: NetworkConfig, k: jax.Array) -> jax.Array:
        """Evaluate one sub-config; return per-station rates (n_stas,) in Mb/s."""
        tx, tx_p, mcs = to_arrays(config)
        result    = scenario(k, tx, tx_p, mcs, return_internals=True)
        internals = result[2]

        # ap_rate[i] = throughput of AP i's transmission in Mb/s (0 if AP inactive)
        ap_rate = internals.average_data_rate[ap_ids_jnp] / jnp.float32(1e6)  # (n_aps,)

        # For each (ap, sta) slot, rate = ap_rate[ap] if that slot is selected, else 0
        ap_rate_expanded = jnp.repeat(ap_rate, max_stas)               # (n_aps * max_stas,)
        selected_flat    = config.selected.ravel().astype(jnp.float32)  # (n_aps * max_stas,)
        rate_flat        = selected_flat * ap_rate_expanded             # (n_aps * max_stas,)

        return rate_flat[valid_flat_idx]  # (n_stas,)

    @jax.jit
    def _evaluate(sol: FSolution, key: jax.Array):
        keys = jax.random.split(key, max_configs)
        # Parallel evaluation of all sub-configs — vmap, not threads
        sub_sta_rates = jax.vmap(_eval_one)(sol.configs, keys)  # (max_configs, n_stas)

        # Apply weight strategy (identity by default; LP ablation replaces this)
        logits_for_eval = weight_strategy(sol, sub_sta_rates)
        weights = jax.nn.softmax(logits_for_eval)                # (max_configs,)

        per_sta = (weights[:, None] * sub_sta_rates).sum(axis=0) # (n_stas,)
        return per_sta, sub_sta_rates, weights

    return _evaluate


# ---------------------------------------------------------------------------
# Leximin comparison
# ---------------------------------------------------------------------------

@jax.jit
def leximin_delta(new_per_sta: jax.Array, old_per_sta: jax.Array) -> jax.Array:
    """Lexicographic max-min delta: delta at the first index where sorted vectors differ.

    Both vectors are sorted ascending (worst station first). The returned scalar is
    new_sorted[k] - old_sorted[k] at the smallest k where they differ.
    If identical, returns 0.0 (SA will accept — promotes exploration).
    """
    a = jnp.sort(new_per_sta)           # (n_stas,) ascending
    b = jnp.sort(old_per_sta)
    diff    = a - b                     # (n_stas,)
    tol     = jnp.float32(1e-6)
    differs = jnp.abs(diff) > tol       # (n_stas,) bool
    first_k = jnp.argmax(differs)       # first True; 0 if all False
    return jnp.where(jnp.any(differs), diff[first_k], jnp.float32(0.0))


def leximin_score(per_sta: jax.Array) -> jax.Array:
    """Cheap scalar summary of a per-station vector for tracking 'best so far'.

    Not used as the acceptance criterion; only for FResult history fields.
    """
    return jnp.min(per_sta) * jnp.float32(1e3) + jnp.sum(per_sta)


# ---------------------------------------------------------------------------
# Neighbourhood operator
# ---------------------------------------------------------------------------

def make_neighbor_f(
    info:        ScenarioInfo,
    max_configs: int,
) -> Callable[[FSolution, jax.Array], FSolution]:
    """Return a JIT-compiled neighbour generator for FSolution.

    Four move types dispatched by jax.lax.switch:
      0 modify_config   — mutate one sub-config's NetworkConfig (prob ∝ n_live)
      1 perturb_weights — add Gaussian noise to all logits        (prob ∝ 1)
      2 add_config      — revive the least-weight slot             (prob ∝ 1)
      3 remove_config   — zero out a live slot                     (prob ∝ 1, 0 if n_live≤1)

    modify_config probability is proportional to n_live (neighbourhood size);
    all other moves have weight 1. No additional bias multiplier.
    A coverage rejection loop (up to REJECTION_CAP tries) discards neighbours that
    leave any station unserved.
    """
    valid_mask  = jnp.array(info.valid_mask, dtype=jnp.int32)
    all_covered = make_coverage_check(info)
    random_config_fn = make_random_config(info)

    @jax.jit
    def _neighbor(sol: FSolution, key: jax.Array) -> FSolution:

        def _sample_candidate(key: jax.Array) -> FSolution:
            weights  = jax.nn.softmax(sol.logits)              # (M,)
            n_live   = jnp.sum(weights > LIVE_EPS).astype(jnp.float32)

            key, op_key, slot_key, val_key, cfg_key = jax.random.split(key, 5)

            # --- Move probability weights ---
            c_modify  = n_live
            c_perturb = jnp.float32(1.0)
            c_add     = jnp.float32(1.0)
            c_remove  = jnp.where(n_live > jnp.float32(1.0), jnp.float32(1.0), jnp.float32(0.0))
            move_logits = jnp.log(jnp.stack([c_modify, c_perturb, c_add, c_remove]) + jnp.float32(1e-9))
            op = jax.random.categorical(op_key, move_logits)

            # --- 0: modify_config ---
            def modify_config(_):
                chosen = jax.random.choice(slot_key, max_configs, p=weights)
                sub = NetworkConfig(
                    sol.configs.selected[chosen],
                    sol.configs.tx_power[chosen],
                    sol.configs.mcs[chosen],
                )
                new_sub = neighbor(sub, val_key, valid_mask)
                new_sel = sol.configs.selected.at[chosen].set(new_sub.selected)
                new_tp  = sol.configs.tx_power.at[chosen].set(new_sub.tx_power)
                new_mcs = sol.configs.mcs.at[chosen].set(new_sub.mcs)
                return FSolution(
                    configs=NetworkConfig(new_sel, new_tp, new_mcs),
                    logits=sol.logits,
                )

            # --- 1: perturb_weights ---
            def perturb_weights(_):
                noise = jax.random.normal(val_key, (max_configs,)) * jnp.float32(PERTURB_SIGMA)
                return FSolution(configs=sol.configs, logits=sol.logits + noise)

            # --- 2: add_config ---
            def add_config(_):
                weakest   = jnp.argmin(weights)
                new_cfg   = random_config_fn(cfg_key)
                new_sel   = sol.configs.selected.at[weakest].set(new_cfg.selected)
                new_tp    = sol.configs.tx_power.at[weakest].set(new_cfg.tx_power)
                new_mcs   = sol.configs.mcs.at[weakest].set(new_cfg.mcs)
                # Set logit so the new slot joins at ~half the current max weight
                new_logit = jnp.max(sol.logits) + jnp.float32(REVIVE_LOGIT_DELTA)
                new_logits = sol.logits.at[weakest].set(new_logit)
                return FSolution(
                    configs=NetworkConfig(new_sel, new_tp, new_mcs),
                    logits=new_logits,
                )

            # --- 3: remove_config ---
            def remove_config(_):
                chosen     = jax.random.choice(slot_key, max_configs, p=weights)
                new_logits = sol.logits.at[chosen].set(jnp.float32(REMOVE_LOGIT))
                return FSolution(configs=sol.configs, logits=new_logits)

            return jax.lax.switch(op, [modify_config, perturb_weights, add_config, remove_config], None)

        # Coverage rejection loop — up to REJECTION_CAP tries.
        # If all retries are exhausted, fall back to the current (valid) sol so
        # the SA step sees delta=0 and accepts (harmless exploration no-op).
        def _cond(state):
            candidate, _, tries = state
            return (~all_covered(candidate)) & (tries < REJECTION_CAP)

        def _body(state):
            _, key, tries = state
            key, sub_key = jax.random.split(key)
            candidate = _sample_candidate(sub_key)
            return candidate, key, tries + 1

        key, init_key = jax.random.split(key)
        init_candidate = _sample_candidate(init_key)
        candidate, _, _ = jax.lax.while_loop(_cond, _body, (init_candidate, key, jnp.int32(1)))
        return jax.lax.cond(all_covered(candidate), lambda: candidate, lambda: sol)

    return _neighbor


# ---------------------------------------------------------------------------
# Shared setup for F-family algorithms
# ---------------------------------------------------------------------------

def setup_f(scenario, seed: int, max_configs: int, weight_strategy):
    """Build info, split keys, generate a random covered initial solution + evaluate it.

    max_configs is auto-bumped to n_stas when the default would be too small to fit
    the required basis configs. JAX will recompile for each distinct effective value.
    """
    info    = make_scenario_info(scenario)
    n_stas  = int(info.valid_mask.sum())
    max_configs = max(max_configs, n_stas)
    evaluate = make_evaluator(scenario, info, max_configs, weight_strategy)

    key                         = jax.random.PRNGKey(seed)
    key, sol_key, eval_key, run_key = jax.random.split(key, 4)

    initial_sol = make_random_f_solution(info, max_configs, sol_key)
    per_sta, _, _ = evaluate(initial_sol, eval_key)

    return info, evaluate, run_key, initial_sol, per_sta, max_configs
