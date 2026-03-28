"""
Neighbor generation functions for all 4 SA versions — pure JAX, JIT-compatible.

All functions share the signature:
    neighbor_vN(config, key, T, T_0, valid_mask, ...) -> SAConfig

Extra per-version params (max_jump_mcs, etc.) are Python ints baked in via
functools.partial before being passed into the JIT-compiled SA loop.
"""
from __future__ import annotations

import mapc_sa._env  # noqa: F401

import jax
import jax.numpy as jnp

from mapc_sa.config import SAConfig


def _activate_ap(config: SAConfig, ap_idx: jax.Array, val_key: jax.Array,
                 valid_mask: jax.Array) -> SAConfig:
    """Set exactly one valid STA for ap_idx to selected=1; deselect all others."""
    max_stas = valid_mask.shape[1]
    probs = valid_mask[ap_idx].astype(jnp.float32)
    probs = probs / (probs.sum() + 1e-9)
    sta_idx = jax.random.choice(val_key, max_stas, p=probs)
    new_row = (
        jnp.zeros(max_stas, dtype=jnp.int32).at[sta_idx].set(1)
        * valid_mask[ap_idx]
    )
    new_sel = config.selected.at[ap_idx].set(new_row)
    return SAConfig(new_sel, config.tx_power, config.mcs)


def _deactivate_ap(config: SAConfig, ap_idx: jax.Array) -> SAConfig:
    """Deselect all STAs for ap_idx. No-op if ap_idx is the last active AP."""
    max_stas = config.selected.shape[1]
    count_active = jnp.sum(config.selected)
    new_sel = jax.lax.cond(
        count_active > 1,
        lambda: config.selected.at[ap_idx].set(jnp.zeros(max_stas, dtype=jnp.int32)),
        lambda: config.selected,
    )
    return SAConfig(new_sel, config.tx_power, config.mcs)


def _mutate_selected_basic(config: SAConfig, ap_idx: jax.Array,
                            key: jax.Array, valid_mask: jax.Array) -> SAConfig:
    """Toggle: if AP active → deactivate; if inactive → activate random STA."""
    key, val_key = jax.random.split(key)
    ap_active = jnp.any(config.selected[ap_idx] > 0)
    return jax.lax.cond(
        ap_active,
        lambda: _deactivate_ap(config, ap_idx),
        lambda: _activate_ap(config, ap_idx, val_key, valid_mask),
    )


def _mutate_tx_power_random(config: SAConfig, ap_idx: jax.Array,
                             sta_idx: jax.Array, key: jax.Array) -> SAConfig:
    new_val = jax.random.randint(key, (), 0, 4)
    new_tp  = config.tx_power.at[ap_idx, sta_idx].set(new_val)
    return SAConfig(config.selected, new_tp, config.mcs)


def _mutate_mcs_random(config: SAConfig, ap_idx: jax.Array,
                        sta_idx: jax.Array, key: jax.Array) -> SAConfig:
    new_val = jax.random.randint(key, (), 0, 14)
    new_mcs = config.mcs.at[ap_idx, sta_idx].set(new_val)
    return SAConfig(config.selected, config.tx_power, new_mcs)


def _mutate_tx_power_ordinal(config: SAConfig, ap_idx: jax.Array,
                              sta_idx: jax.Array, key: jax.Array,
                              max_jump: jax.Array) -> SAConfig:
    delta   = jax.random.randint(key, (), -max_jump, max_jump + 1)
    new_val = jnp.clip(config.tx_power[ap_idx, sta_idx] + delta, 0, 3)
    new_tp  = config.tx_power.at[ap_idx, sta_idx].set(new_val)
    return SAConfig(config.selected, new_tp, config.mcs)


def _mutate_mcs_ordinal(config: SAConfig, ap_idx: jax.Array,
                         sta_idx: jax.Array, key: jax.Array,
                         max_jump: jax.Array) -> SAConfig:
    delta   = jax.random.randint(key, (), -max_jump, max_jump + 1)
    new_val = jnp.clip(config.mcs[ap_idx, sta_idx] + delta, 0, 13)
    new_mcs = config.mcs.at[ap_idx, sta_idx].set(new_val)
    return SAConfig(config.selected, config.tx_power, new_mcs)


def _mutate_selected_concurrency(config: SAConfig, ap_idx: jax.Array,
                                  key: jax.Array, valid_mask: jax.Array,
                                  concurrency_target: float) -> SAConfig:
    """Add or remove an active AP based on sigmoid(target - n_active)."""
    key, add_key, ap_key, val_key = jax.random.split(key, 4)
    n_active = jnp.sum(config.selected).astype(jnp.float32)
    p_add    = jax.nn.sigmoid(jnp.float32(concurrency_target) - n_active)
    do_add   = jax.random.uniform(add_key) < p_add

    n_aps = config.selected.shape[0]

    def add_fn(_):
        is_inactive = ~jnp.any(config.selected > 0, axis=1)
        total       = is_inactive.astype(jnp.float32).sum()
        probs       = jnp.where(total > 0, is_inactive.astype(jnp.float32) / (total + 1e-9),
                                jnp.ones(n_aps) / n_aps)
        target_ap   = jax.random.choice(ap_key, n_aps, p=probs)
        return jax.lax.cond(
            total > 0,
            lambda: _activate_ap(config, target_ap, val_key, valid_mask),
            lambda: config,
        )

    def remove_fn(_):
        is_active = jnp.any(config.selected > 0, axis=1)
        probs     = is_active.astype(jnp.float32)
        total     = probs.sum()
        probs     = jnp.where(total > 0, probs / total, jnp.ones(n_aps) / n_aps)
        target_ap = jax.random.choice(ap_key, n_aps, p=probs)
        return _deactivate_ap(config, target_ap)

    return jax.lax.cond(do_add, add_fn, remove_fn, None)


def neighbor_v1(config: SAConfig, key: jax.Array,
                T: jax.Array, T_0: float,
                valid_mask: jax.Array) -> SAConfig:
    """Version 1: pure SA — random draw from all valid values."""
    n_aps, max_stas = valid_mask.shape
    key, param_key, ap_key, sta_key, val_key = jax.random.split(key, 5)

    param_type = jax.random.randint(param_key, (), 0, 3)
    ap_idx     = jax.random.randint(ap_key,    (), 0, n_aps)
    sta_idx    = jax.random.randint(sta_key,   (), 0, max_stas)

    return jax.lax.switch(param_type, [
        lambda _: _mutate_selected_basic(config, ap_idx, val_key, valid_mask),
        lambda _: _mutate_tx_power_random(config, ap_idx, sta_idx, val_key),
        lambda _: _mutate_mcs_random(config, ap_idx, sta_idx, val_key),
    ], None)


def neighbor_v2(config: SAConfig, key: jax.Array,
                T: jax.Array, T_0: float,
                valid_mask: jax.Array) -> SAConfig:
    """Version 2: ordinal-aware — tx_power and mcs change by ±1 only."""
    n_aps, max_stas = valid_mask.shape
    key, param_key, ap_key, sta_key, val_key = jax.random.split(key, 5)

    param_type = jax.random.randint(param_key, (), 0, 3)
    ap_idx     = jax.random.randint(ap_key,    (), 0, n_aps)
    sta_idx    = jax.random.randint(sta_key,   (), 0, max_stas)

    return jax.lax.switch(param_type, [
        lambda _: _mutate_selected_basic(config, ap_idx, val_key, valid_mask),
        lambda _: _mutate_tx_power_ordinal(config, ap_idx, sta_idx, val_key, jnp.int32(1)),
        lambda _: _mutate_mcs_ordinal(config, ap_idx, sta_idx, val_key, jnp.int32(1)),
    ], None)


def neighbor_v3(config: SAConfig, key: jax.Array,
                T: jax.Array, T_0: float,
                valid_mask: jax.Array,
                max_jump_mcs: int = 7,
                max_jump_power: int = 3) -> SAConfig:
    """Version 3: temperature-dependent jump sizes for ordinal params."""
    n_aps, max_stas = valid_mask.shape
    key, param_key, ap_key, sta_key, val_key = jax.random.split(key, 5)

    t_ratio       = T / jnp.maximum(jnp.float32(T_0), jnp.float32(1e-300))
    eff_max_mcs   = jnp.maximum(1, jnp.round(jnp.float32(max_jump_mcs)   * t_ratio).astype(jnp.int32))
    eff_max_power = jnp.maximum(1, jnp.round(jnp.float32(max_jump_power) * t_ratio).astype(jnp.int32))

    param_type = jax.random.randint(param_key, (), 0, 3)
    ap_idx     = jax.random.randint(ap_key,    (), 0, n_aps)
    sta_idx    = jax.random.randint(sta_key,   (), 0, max_stas)

    return jax.lax.switch(param_type, [
        lambda _: _mutate_selected_basic(config, ap_idx, val_key, valid_mask),
        lambda _: _mutate_tx_power_ordinal(config, ap_idx, sta_idx, val_key, eff_max_power),
        lambda _: _mutate_mcs_ordinal(config, ap_idx, sta_idx, val_key, eff_max_mcs),
    ], None)


def neighbor_v4(config: SAConfig, key: jax.Array,
                T: jax.Array, T_0: float,
                valid_mask: jax.Array,
                max_jump_mcs: int = 7,
                max_jump_power: int = 3,
                concurrency_target: float = 4.0) -> SAConfig:
    """Version 4: concurrency-controlled SA with temperature-dependent jumps."""
    n_aps, max_stas = valid_mask.shape
    key, param_key, ap_key, sta_key, val_key = jax.random.split(key, 5)

    t_ratio       = T / jnp.maximum(jnp.float32(T_0), jnp.float32(1e-300))
    eff_max_mcs   = jnp.maximum(1, jnp.round(jnp.float32(max_jump_mcs)   * t_ratio).astype(jnp.int32))
    eff_max_power = jnp.maximum(1, jnp.round(jnp.float32(max_jump_power) * t_ratio).astype(jnp.int32))

    param_type = jax.random.randint(param_key, (), 0, 3)
    ap_idx     = jax.random.randint(ap_key,    (), 0, n_aps)
    sta_idx    = jax.random.randint(sta_key,   (), 0, max_stas)

    return jax.lax.switch(param_type, [
        lambda _: _mutate_selected_concurrency(config, ap_idx, val_key, valid_mask, concurrency_target),
        lambda _: _mutate_tx_power_ordinal(config, ap_idx, sta_idx, val_key, eff_max_power),
        lambda _: _mutate_mcs_ordinal(config, ap_idx, sta_idx, val_key, eff_max_mcs),
    ], None)
