from __future__ import annotations

import mapc_mh.env  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass

from mapc_mh.config import (
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
# Neighbor generation
# ---------------------------------------------------------------------------
#
# Mutation types:
#   0 - mutate_selected: change AP-STA assignment
#       - If AP inactive: activate with random valid STA
#       - If AP active: 50% deactivate (if >1 active), 50% switch STA (if >1 valid)
#   1 - mutate_tx_power: change tx_power for a random active AP at its selected STA
#   2 - mutate_mcs: change MCS for a random active AP at its selected STA
#
# Invariants maintained:
#   - At most 1 selected STA per AP (row-sum <= 1)
#   - At least 1 AP is active globally (total selected >= 1)
#   - Only valid STAs can be selected (respects valid_mask)
#   - tx_power/mcs mutations always target meaningful positions
# ---------------------------------------------------------------------------


def neighbor(config: NetworkConfig, key: jax.Array, valid_mask: jax.Array) -> NetworkConfig:
    """Generate a neighbor by randomly mutating selected, tx_power, or MCS.

    All mutations are meaningful:
    - Selected mutations can activate, deactivate, or switch STAs
    - tx_power/mcs mutations target active APs at their selected STA position
    """
    n_aps, max_stas = valid_mask.shape
    key, param_key, ap_key, action_key, val_key = jax.random.split(key, 5)

    param_type = jax.random.randint(param_key, (), 0, 3)

    # Precompute useful info about current config
    ap_is_active = jnp.any(config.selected > 0, axis=1)  # (n_aps,) bool
    n_active = jnp.sum(ap_is_active)
    selected_sta_idx = jnp.argmax(config.selected, axis=1)  # (n_aps,) STA index per AP
    n_valid_per_ap = jnp.sum(valid_mask, axis=1)  # (n_aps,)

    # --- Mutation 0: mutate_selected ---
    def mutate_selected(_):
        # Pick a random AP
        ap_idx = jax.random.randint(ap_key, (), 0, n_aps)
        is_active = ap_is_active[ap_idx]
        current_sta = selected_sta_idx[ap_idx]
        n_valid = n_valid_per_ap[ap_idx]

        def _activate(ap_idx):
            """Activate inactive AP with a random valid STA."""
            probs = valid_mask[ap_idx].astype(jnp.float32)
            probs = probs / (probs.sum() + 1e-9)
            new_sta = jax.random.choice(val_key, max_stas, p=probs)
            new_row = jnp.zeros(max_stas, dtype=jnp.int32).at[new_sta].set(1)
            return NetworkConfig(
                config.selected.at[ap_idx].set(new_row),
                config.tx_power,
                config.mcs,
            )

        def _deactivate(ap_idx):
            """Deactivate AP (only if more than 1 AP is active)."""
            new_selected = jax.lax.cond(
                n_active > 1,
                lambda: config.selected.at[ap_idx].set(jnp.zeros(max_stas, dtype=jnp.int32)),
                lambda: config.selected,  # no-op: can't deactivate last AP
            )
            return NetworkConfig(new_selected, config.tx_power, config.mcs)

        def _switch_sta(ap_idx, current_sta):
            """Switch active AP to a different valid STA (only if >1 valid STA)."""
            # Exclude current STA from selection
            probs = valid_mask[ap_idx].astype(jnp.float32)
            probs = probs.at[current_sta].set(0.0)
            probs = probs / (probs.sum() + 1e-9)
            new_sta = jax.random.choice(val_key, max_stas, p=probs)
            new_row = jnp.zeros(max_stas, dtype=jnp.int32).at[new_sta].set(1)

            new_selected = jax.lax.cond(
                n_valid > 1,
                lambda: config.selected.at[ap_idx].set(new_row),
                lambda: config.selected,  # no-op: only 1 valid STA, can't switch
            )
            return NetworkConfig(new_selected, config.tx_power, config.mcs)

        def when_active():
            # 50% chance deactivate, 50% chance switch STA
            do_switch = jax.random.randint(action_key, (), 0, 2)
            return jax.lax.cond(
                do_switch == 1,
                lambda: _switch_sta(ap_idx, current_sta),
                lambda: _deactivate(ap_idx),
            )

        def when_inactive():
            return _activate(ap_idx)

        return jax.lax.cond(is_active, when_active, when_inactive)

    # --- Mutation 1: mutate_tx_power ---
    def mutate_tx_power(_):
        # Pick a random ACTIVE AP (weighted by active mask)
        active_probs = ap_is_active.astype(jnp.float32)
        active_probs = active_probs / (active_probs.sum() + 1e-9)
        ap_idx = jax.random.choice(ap_key, n_aps, p=active_probs)

        # Mutate tx_power at the selected STA position (the one that matters)
        sta_idx = selected_sta_idx[ap_idx]
        new_val = jax.random.randint(val_key, (), 0, 4)
        return NetworkConfig(
            config.selected,
            config.tx_power.at[ap_idx, sta_idx].set(new_val),
            config.mcs,
        )

    # --- Mutation 2: mutate_mcs ---
    def mutate_mcs(_):
        # Pick a random ACTIVE AP (weighted by active mask)
        active_probs = ap_is_active.astype(jnp.float32)
        active_probs = active_probs / (active_probs.sum() + 1e-9)
        ap_idx = jax.random.choice(ap_key, n_aps, p=active_probs)

        # Mutate MCS at the selected STA position (the one that matters)
        sta_idx = selected_sta_idx[ap_idx]
        new_val = jax.random.randint(val_key, (), 0, 14)
        return NetworkConfig(
            config.selected,
            config.tx_power,
            config.mcs.at[ap_idx, sta_idx].set(new_val),
        )

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
