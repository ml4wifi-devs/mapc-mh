"""Core primitives for the F-Optimal (fairness) metaheuristic family.

Solution representation
-----------------------
An FSolution holds a fixed-size batch of Co-SR sub-configurations (NetworkConfig)
and a boolean active mask. Weights are computed externally by an LP solver (max-min
over the per-config per-station rate matrix). The metaheuristic proposes configs only;
the LP assigns optimal time fractions.

Evaluation
----------
All sub-configs are evaluated in parallel via jax.vmap over the fixed leading
dimension (JIT-compiled). The resulting per-config per-station rate matrix is then
passed to the LP solver (Python-side, not JIT).

Neighbourhood operators
-----------------------
Three moves (modify_config, add_config, remove_config) dispatched via jax.lax.switch.
perturb_weights is dropped — weights are LP-optimal given the config set, so there is
nothing to tune. Move probabilities are proportional to neighbourhood size.

No threads or processes are used anywhere. Parallelism is jax.vmap only.
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
# Module-level knobs
# ---------------------------------------------------------------------------

LIVE_EPS     = 1e-4   # weight threshold for "active" slot (used only for coverage checks)
MODIFY_BIAS  = 3.0    # multiplier making modify_config more likely per active config
REJECTION_CAP = 32    # maximum retries in the coverage rejection loop


# ---------------------------------------------------------------------------
# Solution type
# ---------------------------------------------------------------------------

class FSolution(NamedTuple):
    """A set of Co-SR sub-configurations for F-Optimal search.

    configs : NetworkConfig — batched over a fixed leading axis of size max_configs.
              Each field (selected, tx_power, mcs) has shape (max_configs, n_aps, max_stas).
    active  : (max_configs,) bool — which slots are currently in use.
              LP weights are computed externally from the per-config rate matrix.
    """
    configs: NetworkConfig
    active:  jax.Array   # (max_configs,) bool


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FResult:
    best_solution:   FSolution
    best_score:      float        # surrogate: min_rate * 1e3 + sum_rate
    best_min_rate:   float        # LP-optimal max-min throughput (Mb/s)
    best_sum_rate:   float        # total throughput (Mb/s)
    best_fairness:   float        # Jain's fairness index over per-station rates
    best_per_sta:    list[float]  # per-station rates of the best solution
    history:         list[float]  # surrogate score per step (best-so-far)
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
        live    = sol.active[:, None, None]                              # (M, 1, 1) bool
        contrib = (sol.configs.selected.astype(jnp.bool_) & live).any(axis=0)  # (n_aps, max_stas)
        return jnp.all(contrib | ~valid_mask)

    return _all_covered


# ---------------------------------------------------------------------------
# Initial solution
# ---------------------------------------------------------------------------

def make_random_f_solution(
    info:        ScenarioInfo,
    max_configs: int,
    key:         jax.Array,
) -> FSolution:
    """Generate an FSolution guaranteed to cover all stations.

    One basis sub-config per station forces that (ap, sta) pair to be assigned.
    Remaining slots are filled with random configs but marked inactive (active=False).
    Only basis configs are active at start; the search adds/removes configs from there.
    """
    random_config_fn = make_random_config(info)
    max_stas = info.max_stas

    valid_pairs = np.argwhere(info.valid_mask)   # (n_stas, 2) numpy
    n_stas      = len(valid_pairs)

    if n_stas > max_configs:
        raise ValueError(
            f'max_configs={max_configs} < n_stas={n_stas}. '
            f'Increase max_configs so every station can have a basis config.'
        )

    @jax.jit
    def _make(key: jax.Array) -> FSolution:
        key, rand_key = jax.random.split(key)
        basis_keys = jax.random.split(key, n_stas)

        def make_basis(ap_idx_sta_slot, bkey):
            ap_idx, sta_slot = ap_idx_sta_slot[0], ap_idx_sta_slot[1]
            base = random_config_fn(bkey)
            forced_row   = jnp.zeros(max_stas, dtype=jnp.int32).at[sta_slot].set(1)
            new_selected = base.selected.at[ap_idx].set(forced_row)
            return NetworkConfig(new_selected, base.tx_power, base.mcs)

        pairs_jnp     = jnp.array(valid_pairs, dtype=jnp.int32)
        basis_configs = jax.vmap(make_basis)(pairs_jnp, basis_keys)

        n_filler   = max_configs - n_stas
        fill_keys  = jax.random.split(rand_key, max(n_filler, 1))
        filler_configs = jax.vmap(random_config_fn)(fill_keys[:max(n_filler, 1)])

        if n_filler > 0:
            all_selected = jnp.concatenate([basis_configs.selected, filler_configs.selected], axis=0)
            all_tp       = jnp.concatenate([basis_configs.tx_power, filler_configs.tx_power], axis=0)
            all_mcs      = jnp.concatenate([basis_configs.mcs,      filler_configs.mcs],      axis=0)
        else:
            all_selected = basis_configs.selected
            all_tp       = basis_configs.tx_power
            all_mcs      = basis_configs.mcs

        configs = NetworkConfig(all_selected, all_tp, all_mcs)

        # Only basis slots are active; fillers are inactive
        active = jnp.concatenate([
            jnp.ones(n_stas,              dtype=jnp.bool_),
            jnp.zeros(max(n_filler, 0),   dtype=jnp.bool_),
        ])
        return FSolution(configs=configs, active=active)

    return _make(key)


# ---------------------------------------------------------------------------
# Evaluator — JIT-compiled, vmap over sub-configs
# ---------------------------------------------------------------------------

def make_evaluator(
    scenario,
    info:        ScenarioInfo,
    max_configs: int,
) -> Callable[[FSolution, jax.Array], jax.Array]:
    """Return a JIT-compiled evaluator: (FSolution, key) -> sub_sta_rates (max_configs, n_stas).

    Runs all sub-configs in parallel via jax.vmap. The returned rate matrix is then
    passed to the LP solver (Python-side) to compute optimal mixing weights.

    Per-station throughput extraction
    ----------------------------------
    average_data_rate is indexed by TRANSMITTER node. For AP ap_idx:
        rate(ap_idx, sta_slot) = selected[ap_idx, sta_slot] * avg_data_rate[ap_node_id] / 1e6
    """
    to_arrays      = make_config_to_arrays(info)
    ap_ids_jnp     = jnp.array(info.ap_ids, dtype=jnp.int32)   # (n_aps,)
    max_stas       = info.max_stas
    valid_flat_idx = jnp.array(np.flatnonzero(info.valid_mask), dtype=jnp.int32)  # (n_stas,)

    def _eval_one(config: NetworkConfig, k: jax.Array) -> jax.Array:
        tx, tx_p, mcs = to_arrays(config)
        result    = scenario(k, tx, tx_p, mcs, return_internals=True)
        ap_rate   = result[2].average_data_rate[ap_ids_jnp] / jnp.float32(1e6)  # (n_aps,) Mb/s
        ap_rate_expanded = jnp.repeat(ap_rate, max_stas)                          # (n_aps*max_stas,)
        selected_flat    = config.selected.ravel().astype(jnp.float32)
        rate_flat        = selected_flat * ap_rate_expanded
        return rate_flat[valid_flat_idx]   # (n_stas,)

    @jax.jit
    def _evaluate(sol: FSolution, key: jax.Array) -> jax.Array:
        keys = jax.random.split(key, max_configs)
        return jax.vmap(_eval_one)(sol.configs, keys)   # (max_configs, n_stas)

    return _evaluate


# ---------------------------------------------------------------------------
# Neighbourhood operator
# ---------------------------------------------------------------------------

def make_neighbor_f(
    info:        ScenarioInfo,
    max_configs: int,
    modify_bias: float = MODIFY_BIAS,
) -> Callable[[FSolution, jax.Array], FSolution]:
    """Return a JIT-compiled neighbour generator for FSolution.

    Three move types dispatched by jax.lax.switch:
      0 modify_config  — mutate one active sub-config's NetworkConfig  (prob ∝ n_active * modify_bias)
      1 add_config     — activate an inactive slot with a fresh config  (prob ∝ 1)
      2 remove_config  — deactivate an active slot                      (prob ∝ 1, 0 if n_active≤1)

    A coverage rejection loop (up to REJECTION_CAP tries) discards neighbours that
    leave any station uncovered. Falls back to current sol on cap exhaustion.
    """
    valid_mask   = jnp.array(info.valid_mask, dtype=jnp.int32)
    all_covered  = make_coverage_check(info)
    random_config_fn = make_random_config(info)

    @jax.jit
    def _neighbor(sol: FSolution, key: jax.Array) -> FSolution:

        def _sample_candidate(key: jax.Array) -> FSolution:
            active_f = sol.active.astype(jnp.float32)   # (M,)
            n_active = jnp.sum(active_f)

            key, op_key, slot_key, val_key, cfg_key = jax.random.split(key, 5)

            # Move probabilities proportional to neighbourhood size
            c_modify = modify_bias * n_active
            c_add    = jnp.float32(1.0)
            c_remove = jnp.where(n_active > jnp.float32(1.0), jnp.float32(1.0), jnp.float32(0.0))
            move_logits = jnp.log(jnp.stack([c_modify, c_add, c_remove]) + jnp.float32(1e-9))
            op = jax.random.categorical(op_key, move_logits)

            # --- 0: modify_config ---
            def modify_config(_):
                probs  = active_f / jnp.maximum(n_active, jnp.float32(1.0))
                chosen = jax.random.choice(slot_key, max_configs, p=probs)
                sub = NetworkConfig(
                    sol.configs.selected[chosen],
                    sol.configs.tx_power[chosen],
                    sol.configs.mcs[chosen],
                )
                new_sub = neighbor(sub, val_key, valid_mask)
                new_sel = sol.configs.selected.at[chosen].set(new_sub.selected)
                new_tp  = sol.configs.tx_power.at[chosen].set(new_sub.tx_power)
                new_mcs = sol.configs.mcs.at[chosen].set(new_sub.mcs)
                return FSolution(NetworkConfig(new_sel, new_tp, new_mcs), sol.active)

            # --- 1: add_config ---
            def add_config(_):
                # First inactive slot (argmax on ~active)
                weakest = jnp.argmax((~sol.active).astype(jnp.int32))
                new_cfg = random_config_fn(cfg_key)
                new_sel = sol.configs.selected.at[weakest].set(new_cfg.selected)
                new_tp  = sol.configs.tx_power.at[weakest].set(new_cfg.tx_power)
                new_mcs = sol.configs.mcs.at[weakest].set(new_cfg.mcs)
                new_active = sol.active.at[weakest].set(True)
                return FSolution(NetworkConfig(new_sel, new_tp, new_mcs), new_active)

            # --- 2: remove_config ---
            def remove_config(_):
                probs  = active_f / jnp.maximum(n_active, jnp.float32(1.0))
                chosen = jax.random.choice(slot_key, max_configs, p=probs)
                return FSolution(sol.configs, sol.active.at[chosen].set(False))

            return jax.lax.switch(op, [modify_config, add_config, remove_config], None)

        # Coverage rejection loop
        def _cond(state):
            candidate, _, tries = state
            return (~all_covered(candidate)) & (tries < REJECTION_CAP)

        def _body(state):
            _, key, tries = state
            key, sub_key = jax.random.split(key)
            return _sample_candidate(sub_key), key, tries + jnp.int32(1)

        key, init_key = jax.random.split(key)
        init_candidate = _sample_candidate(init_key)
        candidate, _, _ = jax.lax.while_loop(_cond, _body, (init_candidate, key, jnp.int32(1)))
        return jax.lax.cond(all_covered(candidate), lambda: candidate, lambda: sol)

    return _neighbor


# ---------------------------------------------------------------------------
# Shared setup for F-family algorithms
# ---------------------------------------------------------------------------

def setup_f(scenario, seed: int, max_configs: int):
    """Build info, split keys, generate a covered initial solution and evaluate it.

    Returns: (info, evaluate, run_key, initial_sol, initial_sub_sta_rates, max_configs)
    max_configs is auto-bumped to n_stas when needed.
    """
    info    = make_scenario_info(scenario)
    n_stas  = int(info.valid_mask.sum())
    max_configs = max(max_configs, n_stas)
    evaluate = make_evaluator(scenario, info, max_configs)

    key                             = jax.random.PRNGKey(seed)
    key, sol_key, eval_key, run_key = jax.random.split(key, 4)

    initial_sol       = make_random_f_solution(info, max_configs, sol_key)
    initial_sub_rates = evaluate(initial_sol, eval_key)

    return info, evaluate, run_key, initial_sol, initial_sub_rates, max_configs
