"""Simulated Annealing (SA)."""
from __future__ import annotations

import mapc_sa.env  # noqa: F401

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mapc_sa.config import NetworkConfig, ScenarioInfo, make_config_to_arrays, make_random_config
from mapc_sa.methods.core import _build_result, _setup, _update_top_n, neighbor
from mapc_sa.methods.core import Result


class _SAState(NamedTuple):
    config:       NetworkConfig
    current_rate: jax.Array
    best_config:  NetworkConfig
    best_rate:    jax.Array
    top_rates:    jax.Array
    top_configs:  NetworkConfig
    key:          jax.Array


def _calibrate_T0(
    scenario,
    info:              ScenarioInfo,
    seed:              int   = 42,
    n_steps:           int   = 200,
    target_acceptance: float = 0.80,
) -> float:
    """Binary-search for T_0 that achieves ~target_acceptance on the first n_steps."""
    to_arrays     = make_config_to_arrays(info)
    random_config = make_random_config(info)
    valid_mask    = jnp.array(info.valid_mask, dtype=jnp.int32)

    key          = jax.random.PRNGKey(seed)
    key, cfg_key = jax.random.split(key)
    current      = random_config(cfg_key)

    key, sim_key  = jax.random.split(key)
    tx, tx_p, mcs = to_arrays(current)
    current_rate  = float(scenario(sim_key, tx, tx_p, mcs, return_internals=True)[0])

    @jax.jit
    def _step(key, current):
        key, nbr_key, sim_key = jax.random.split(key, 3)
        candidate         = neighbor(current, nbr_key, valid_mask)
        tx, tx_p, mcs     = to_arrays(candidate)
        rate, *_          = scenario(sim_key, tx, tx_p, mcs)
        return key, candidate, rate

    deltas: list[float] = []
    for _ in range(n_steps):
        key, current, candidate_rate = _step(key, current)
        deltas.append(float(candidate_rate) - current_rate)
        current_rate = float(candidate_rate)

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


def _make_runner(scenario, info: ScenarioInfo, T_0: float, T_decay: float, n_steps: int, top_n: int):
    to_arrays  = make_config_to_arrays(info)
    valid_mask = jnp.array(info.valid_mask, dtype=jnp.int32)
    n_aps      = info.n_aps
    max_stas   = info.max_stas
    T_0_f      = jnp.float32(T_0)
    T_decay_f  = jnp.float32(T_decay)

    def _evaluate(config: NetworkConfig, key: jax.Array) -> jax.Array:
        tx, tx_p, mcs = to_arrays(config)
        return scenario(key, tx, tx_p, mcs, return_internals=True)[0].astype(jnp.float32)

    def _step(state: _SAState, step_idx: jax.Array):
        T   = T_0_f * (T_decay_f ** step_idx.astype(jnp.float32))
        key, nbr_key, sim_key, acc_key = jax.random.split(state.key, 4)

        candidate      = neighbor(state.config, nbr_key, valid_mask)
        candidate_rate = _evaluate(candidate, sim_key)

        delta      = candidate_rate - state.current_rate
        log_accept = jnp.minimum(jnp.float32(0.0), delta / jnp.maximum(T, jnp.float32(1e-300)))
        accept     = jnp.log(jax.random.uniform(acc_key)) < log_accept

        new_config = jax.lax.cond(accept, lambda: candidate,     lambda: state.config)
        new_rate   = jnp.where(accept,    candidate_rate,        state.current_rate)

        new_best_config, new_best_rate = jax.lax.cond(
            new_rate > state.best_rate,
            lambda: (new_config, new_rate),
            lambda: (state.best_config, state.best_rate),
        )
        new_top_rates, new_top_configs = _update_top_n(
            state.top_rates, state.top_configs, new_rate, new_config,
        )
        return _SAState(
            config       = new_config,
            current_rate = new_rate,
            best_config  = new_best_config,
            best_rate    = new_best_rate,
            top_rates    = new_top_rates,
            top_configs  = new_top_configs,
            key          = key,
        ), (new_rate, new_best_rate)

    @jax.jit
    def _run(key, initial_config, initial_rate):
        init_state = _SAState(
            config       = initial_config,
            current_rate = initial_rate,
            best_config  = initial_config,
            best_rate    = initial_rate,
            top_rates    = jnp.full((top_n,), -jnp.inf, dtype=jnp.float32),
            top_configs  = NetworkConfig(
                selected = jnp.zeros((top_n, n_aps, max_stas), dtype=jnp.int32),
                tx_power = jnp.zeros((top_n, n_aps, max_stas), dtype=jnp.int32),
                mcs      = jnp.zeros((top_n, n_aps, max_stas), dtype=jnp.int32),
            ),
            key          = key,
        )
        return jax.lax.scan(_step, init_state, jnp.arange(n_steps, dtype=jnp.int32))

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
    seed:    int   = 42,
    n_steps: int   = 2000,
    top_n:   int   = 10,
    T_decay: float = 0.999,
    T_0:     float | None = None,
) -> Result:
    """Run Simulated Annealing on *scenario*.

    T_0 is calibrated automatically if not provided.
    """
    info, run_key, initial_config, initial_rate = _setup(scenario, seed)

    if T_0 is None:
        T_0 = _calibrate_T0(scenario, info, seed=seed)

    runner = _make_runner(scenario, info, T_0, T_decay, n_steps, top_n)
    return runner(run_key, initial_config, initial_rate)
