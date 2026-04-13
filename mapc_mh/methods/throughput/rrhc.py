"""Random Restart Hill Climbing (RRHC)."""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mapc_mh.config import NetworkConfig, make_config_to_arrays, make_random_config
from mapc_mh.methods.core import _build_result, _setup, _update_top_n, neighbor
from mapc_mh.methods.core import Result


class _RRHCState(NamedTuple):
    config:           NetworkConfig
    current_rate:     jax.Array
    best_config:      NetworkConfig
    best_rate:        jax.Array
    top_rates:        jax.Array
    top_configs:      NetworkConfig
    steps_no_improve: jax.Array
    key:              jax.Array


def _make_runner(scenario, info, n_steps: int, top_n: int, restart_threshold: int):
    to_arrays     = make_config_to_arrays(info)
    random_config = make_random_config(info)
    valid_mask    = jnp.array(info.valid_mask, dtype=jnp.int32)
    n_aps         = info.n_aps
    max_stas      = info.max_stas

    def _evaluate(config: NetworkConfig, key: jax.Array) -> jax.Array:
        tx, tx_p, mcs = to_arrays(config)
        return scenario(key, tx, tx_p, mcs, return_internals=True)[0].astype(jnp.float32)

    def _step(state: _RRHCState, _):
        key, nbr_key, sim_key, rst_key, rst_sim_key = jax.random.split(state.key, 5)

        candidate      = neighbor(state.config, nbr_key, valid_mask)
        candidate_rate = _evaluate(candidate, sim_key)

        accept     = candidate_rate > state.current_rate
        new_config = jax.lax.cond(accept, lambda: candidate,    lambda: state.config)
        new_rate   = jnp.where(accept,    candidate_rate,       state.current_rate)
        new_no_imp = jnp.where(accept,    jnp.int32(0),         state.steps_no_improve + 1)

        should_restart = new_no_imp >= restart_threshold
        rst_config     = random_config(rst_key)
        rst_rate       = _evaluate(rst_config, rst_sim_key)
        new_config     = jax.lax.cond(should_restart, lambda: rst_config, lambda: new_config)
        new_rate       = jnp.where(should_restart, rst_rate, new_rate)
        new_no_imp     = jnp.where(should_restart, jnp.int32(0), new_no_imp)

        new_best_config, new_best_rate = jax.lax.cond(
            new_rate > state.best_rate,
            lambda: (new_config, new_rate),
            lambda: (state.best_config, state.best_rate),
        )
        new_top_rates, new_top_configs = _update_top_n(
            state.top_rates, state.top_configs, new_rate, new_config,
        )
        return _RRHCState(
            config           = new_config,
            current_rate     = new_rate,
            best_config      = new_best_config,
            best_rate        = new_best_rate,
            top_rates        = new_top_rates,
            top_configs      = new_top_configs,
            steps_no_improve = new_no_imp,
            key              = key,
        ), (new_rate, new_best_rate)

    @jax.jit
    def _run(key, initial_config, initial_rate):
        init_state = _RRHCState(
            config           = initial_config,
            current_rate     = initial_rate,
            best_config      = initial_config,
            best_rate        = initial_rate,
            top_rates        = jnp.full((top_n,), -jnp.inf, dtype=jnp.float32),
            top_configs      = NetworkConfig(
                selected = jnp.zeros((top_n, n_aps, max_stas), dtype=jnp.int32),
                tx_power = jnp.zeros((top_n, n_aps, max_stas), dtype=jnp.int32),
                mcs      = jnp.zeros((top_n, n_aps, max_stas), dtype=jnp.int32),
            ),
            steps_no_improve = jnp.int32(0),
            key              = key,
        )
        return jax.lax.scan(_step, init_state, None, length=n_steps)

    def runner(key, initial_config, initial_rate):
        final_state, (history, best_history) = _run(key, initial_config, initial_rate)
        return _build_result(
            final_state.best_config, final_state.best_rate,
            final_state.top_rates,   final_state.top_configs,
            history, best_history, top_n, info,
        )

    return runner


def run(
    scenario,
    *,
    seed:              int = 42,
    n_steps:           int = 2000,
    top_n:             int = 10,
    restart_threshold: int = 100,
) -> Result:
    """Run Random Restart Hill Climbing on *scenario*."""
    info, run_key, initial_config, initial_rate = _setup(scenario, seed)
    runner = _make_runner(scenario, info, n_steps, top_n, restart_threshold)
    return runner(run_key, initial_config, initial_rate)
