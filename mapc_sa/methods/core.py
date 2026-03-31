from __future__ import annotations

import mapc_sa.env  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass

from mapc_sa.config import (
    NetworkConfig, ScenarioInfo,
    make_config_to_arrays, make_random_config, make_scenario_info,
)


@dataclass
class Result:
    best_config  : NetworkConfig
    best_rate    : float
    top_configs  : list[tuple[float, NetworkConfig]]
    history      : list[float]
    best_history : list[float]
    info         : ScenarioInfo | None = None


# ---------------------------------------------------------------------------
# Neighbor generation (random draw from all valid values)
# ---------------------------------------------------------------------------

def _activate_ap(
    config: NetworkConfig, ap_idx: jax.Array,
    val_key: jax.Array, valid_mask: jax.Array,
) -> NetworkConfig:
    max_stas = valid_mask.shape[1]
    probs    = valid_mask[ap_idx].astype(jnp.float32)
    probs    = probs / (probs.sum() + 1e-9)
    sta_idx  = jax.random.choice(val_key, max_stas, p=probs)
    new_row  = (
        jnp.zeros(max_stas, dtype=jnp.int32).at[sta_idx].set(1)
        * valid_mask[ap_idx]
    )
    return NetworkConfig(config.selected.at[ap_idx].set(new_row), config.tx_power, config.mcs)


def _deactivate_ap(config: NetworkConfig, ap_idx: jax.Array) -> NetworkConfig:
    max_stas = config.selected.shape[1]
    new_sel  = jax.lax.cond(
        jnp.sum(config.selected) > 1,
        lambda: config.selected.at[ap_idx].set(jnp.zeros(max_stas, dtype=jnp.int32)),
        lambda: config.selected,
    )
    return NetworkConfig(new_sel, config.tx_power, config.mcs)


def neighbor(config: NetworkConfig, key: jax.Array, valid_mask: jax.Array) -> NetworkConfig:
    """Generate a neighbor: randomly mutate selected AP, tx_power, or MCS."""
    n_aps, max_stas = valid_mask.shape
    key, param_key, ap_key, sta_key, val_key = jax.random.split(key, 5)

    param_type = jax.random.randint(param_key, (), 0, 3)
    ap_idx     = jax.random.randint(ap_key,    (), 0, n_aps)
    sta_idx    = jax.random.randint(sta_key,   (), 0, max_stas)

    def mutate_selected(_):
        ap_active = jnp.any(config.selected[ap_idx] > 0)
        return jax.lax.cond(
            ap_active,
            lambda: _deactivate_ap(config, ap_idx),
            lambda: _activate_ap(config, ap_idx, val_key, valid_mask),
        )

    def mutate_tx_power(_):
        new_val = jax.random.randint(val_key, (), 0, 4)
        return NetworkConfig(config.selected, config.tx_power.at[ap_idx, sta_idx].set(new_val), config.mcs)

    def mutate_mcs(_):
        new_val = jax.random.randint(val_key, (), 0, 14)
        return NetworkConfig(config.selected, config.tx_power, config.mcs.at[ap_idx, sta_idx].set(new_val))

    return jax.lax.switch(param_type, [mutate_selected, mutate_tx_power, mutate_mcs], None)


# ---------------------------------------------------------------------------
# Top-N buffer (shared by all methods)
# ---------------------------------------------------------------------------

def _update_top_n(
    top_rates: jax.Array, top_configs: NetworkConfig,
    new_rate:  jax.Array, new_config:  NetworkConfig,
) -> tuple[jax.Array, NetworkConfig]:
    min_idx       = jnp.argmin(top_rates)
    should_insert = new_rate > top_rates[min_idx]

    new_top_rates = jax.lax.cond(should_insert, lambda: top_rates.at[min_idx].set(new_rate),                   lambda: top_rates)
    new_top_sel   = jax.lax.cond(should_insert, lambda: top_configs.selected.at[min_idx].set(new_config.selected), lambda: top_configs.selected)
    new_top_tp    = jax.lax.cond(should_insert, lambda: top_configs.tx_power.at[min_idx].set(new_config.tx_power), lambda: top_configs.tx_power)
    new_top_mcs   = jax.lax.cond(should_insert, lambda: top_configs.mcs.at[min_idx].set(new_config.mcs),           lambda: top_configs.mcs)

    return new_top_rates, NetworkConfig(new_top_sel, new_top_tp, new_top_mcs)


# ---------------------------------------------------------------------------
# Shared initialization and result construction
# ---------------------------------------------------------------------------

def _setup(scenario, seed: int):
    """Build info, split keys, and generate a random initial config + rate."""
    info          = make_scenario_info(scenario)
    to_arrays     = make_config_to_arrays(info)
    random_config = make_random_config(info)

    key                          = jax.random.PRNGKey(seed)
    key, cfg_key, sim_key, run_key = jax.random.split(key, 4)

    initial_config = random_config(cfg_key)
    tx, tx_p, mcs  = to_arrays(initial_config)
    initial_rate   = scenario(sim_key, tx, tx_p, mcs, return_internals=True)[0].astype(jnp.float32)

    return info, run_key, initial_config, initial_rate


def _build_result(
    best_config:    NetworkConfig,
    best_rate:      jax.Array,
    top_rates:      jax.Array,
    top_configs:    NetworkConfig,
    history_arr:    jax.Array,
    bhistory_arr:   jax.Array,
    top_n:          int,
    info:           ScenarioInfo,
) -> Result:
    order   = jnp.argsort(-top_rates)
    s_rates = top_rates[order]
    s_sel   = top_configs.selected[order]
    s_tp    = top_configs.tx_power[order]
    s_mcs   = top_configs.mcs[order]

    top_list = [
        (float(s_rates[i]), NetworkConfig(s_sel[i], s_tp[i], s_mcs[i]))
        for i in range(top_n)
        if float(s_rates[i]) > -jnp.inf
    ]

    return Result(
        best_config  = best_config,
        best_rate    = float(best_rate),
        top_configs  = top_list,
        history      = [float(r) for r in np.array(history_arr)],
        best_history = [float(r) for r in np.array(bhistory_arr)],
        info         = info,
    )
