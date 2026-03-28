from __future__ import annotations

import mapc_sa._env  # noqa: F401 — must be first

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class SAConfig(NamedTuple):
    """JAX-compatible network configuration.

    selected : (n_aps, max_stas) int32  — 1 if AP transmits to that STA slot
    tx_power : (n_aps, max_stas) int32  — 0-3
    mcs      : (n_aps, max_stas) int32  — 0-13

    Invariants (enforced by neighbor functions):
    - At most 1 selected=1 per AP (row-sum <= 1)
    - At least 1 entry is 1 across the whole selected array
    """
    selected: jax.Array
    tx_power: jax.Array
    mcs: jax.Array


@dataclass
class ScenarioInfo:
    """Fixed per-scenario metadata — Python level, not a JAX pytree."""
    ap_ids:     np.ndarray   # (n_aps,)          int32  AP node IDs
    sta_ids:    np.ndarray   # (n_aps, max_stas)  int32  STA node IDs (0-padded)
    valid_mask: np.ndarray   # (n_aps, max_stas)  int32  1 where STA exists
    n_nodes:    int
    n_aps:      int
    max_stas:   int


def make_scenario_info(scenario) -> ScenarioInfo:
    assoc = scenario.associations
    ap_ids_list = sorted(assoc.keys())
    n_aps = len(ap_ids_list)
    max_stas = max(len(stas) for stas in assoc.values())

    all_ids: set[int] = set(ap_ids_list)
    for stas in assoc.values():
        all_ids.update(stas)
    n_nodes = max(all_ids) + 1

    sta_ids    = np.zeros((n_aps, max_stas), dtype=np.int32)
    valid_mask = np.zeros((n_aps, max_stas), dtype=np.int32)

    for i, ap_id in enumerate(ap_ids_list):
        for j, sta_id in enumerate(assoc[ap_id]):
            sta_ids[i, j]    = sta_id
            valid_mask[i, j] = 1

    return ScenarioInfo(
        ap_ids     = np.array(ap_ids_list, dtype=np.int32),
        sta_ids    = sta_ids,
        valid_mask = valid_mask,
        n_nodes    = n_nodes,
        n_aps      = n_aps,
        max_stas   = max_stas,
    )


def make_config_to_arrays(info: ScenarioInfo):
    """Return a JIT-compiled fn: SAConfig -> (tx, tx_power, mcs) simulator arrays.

    n_nodes is baked in via closure so the output shape is compile-time constant.
    """
    n_nodes  = info.n_nodes
    n_aps    = info.n_aps
    ap_ids   = jnp.array(info.ap_ids,  dtype=jnp.int32)
    sta_ids  = jnp.array(info.sta_ids, dtype=jnp.int32)
    arange   = jnp.arange(n_aps, dtype=jnp.int32)

    @jax.jit
    def _to_arrays(config: SAConfig):
        active      = jnp.any(config.selected > 0, axis=1)           # (n_aps,) bool
        sel_local   = jnp.argmax(config.selected, axis=1)            # (n_aps,) local STA idx
        sel_global  = sta_ids[arange, sel_local]                     # (n_aps,) global STA ID
        sel_tp      = config.tx_power[arange, sel_local]             # (n_aps,)
        sel_mcs     = config.mcs[arange, sel_local]                  # (n_aps,)
        active_i    = active.astype(jnp.int32)

        tx          = jnp.zeros((n_nodes, n_nodes), dtype=jnp.int32)
        tx          = tx.at[ap_ids, sel_global].add(active_i)

        tx_power_arr = jnp.zeros(n_nodes, dtype=jnp.int32)
        tx_power_arr = tx_power_arr.at[ap_ids].set(sel_tp * active_i)

        mcs_arr      = jnp.zeros(n_nodes, dtype=jnp.int32)
        mcs_arr      = mcs_arr.at[ap_ids].set(sel_mcs * active_i)

        return tx, tx_power_arr, mcs_arr

    return _to_arrays


def make_random_config(info: ScenarioInfo):
    """Return a JIT-compiled fn: PRNGKey -> SAConfig."""
    n_aps      = info.n_aps
    max_stas   = info.max_stas
    valid_mask = jnp.array(info.valid_mask, dtype=jnp.float32)

    @jax.jit
    def _random_config(key: jax.Array) -> SAConfig:
        key, k_key, perm_key, aps_key = jax.random.split(key, 4)
        ap_keys = jax.random.split(aps_key, n_aps)

        k    = jax.random.randint(k_key, (), 1, n_aps + 1)
        perm = jax.random.permutation(perm_key, n_aps)
        rank = jnp.argsort(perm)          # rank[ap] = position in permutation
        is_active = rank < k              # (n_aps,) bool

        def gen_ap(ap_key, ap_active, ap_valid):
            k1, k2, k3 = jax.random.split(ap_key, 3)
            probs = ap_valid / (ap_valid.sum() + 1e-9)
            sta_idx = jax.random.choice(k1, max_stas, p=probs)
            sel_row = (
                jnp.zeros(max_stas, dtype=jnp.int32).at[sta_idx].set(1)
                * ap_valid.astype(jnp.int32)
                * ap_active.astype(jnp.int32)
            )
            tp_row  = jax.random.randint(k2, (max_stas,), 0, 4)
            mcs_row = jax.random.randint(k3, (max_stas,), 0, 14)
            return sel_row, tp_row, mcs_row

        selected, tx_power, mcs = jax.vmap(gen_ap)(ap_keys, is_active, valid_mask)
        return SAConfig(selected=selected, tx_power=tx_power, mcs=mcs)

    return _random_config


def config_to_serializable(config: SAConfig, info: ScenarioInfo) -> dict:
    """Convert JAX SAConfig to a JSON-serializable dict."""
    sel = np.array(config.selected).tolist()
    tp  = np.array(config.tx_power).tolist()
    mcs = np.array(config.mcs).tolist()
    return {
        str(info.ap_ids[i]): [
            [sel[i][j], tp[i][j], mcs[i][j]]
            for j in range(info.max_stas)
            if info.valid_mask[i, j]
        ]
        for i in range(info.n_aps)
    }
