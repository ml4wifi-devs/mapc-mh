"""
JIT-compiled simulated annealing loop using jax.lax.scan.

Main entry point: make_sa_runner()
  - Closes over scenario-specific info (n_nodes, AP/STA IDs) so output array
    shapes are compile-time constants.
  - Returns a JIT-compiled fn: (key, initial_config, initial_rate) -> SAResult

T_0 calibration: calibrate_T0()
  - Plain Python loop calling the JIT simulator once per step.
  - Called once before the SA run; not performance-critical.
"""
from __future__ import annotations

import mapc_sa._env  # noqa: F401

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mapc_sa.config import SAConfig, ScenarioInfo, make_config_to_arrays, make_random_config


@dataclass
class SAResult:
    best_config:  SAConfig
    best_rate:    float
    top_configs:  list[tuple[float, SAConfig]]
    history:      list[float]
    T_0:          float
    info:         ScenarioInfo | None = None


class SAState(NamedTuple):
    config:        SAConfig
    current_rate:  jax.Array
    best_config:   SAConfig
    best_rate:     jax.Array
    top_rates:     jax.Array     # (top_n,) float32
    top_configs:   SAConfig      # (top_n, n_aps, max_stas)
    key:           jax.Array


def _update_top_n(
    top_rates: jax.Array, top_configs: SAConfig,
    new_rate:  jax.Array, new_config:  SAConfig,
) -> tuple[jax.Array, SAConfig]:
    """Replace the worst entry if new_rate is better."""
    min_idx       = jnp.argmin(top_rates)
    should_insert = new_rate > top_rates[min_idx]

    new_top_rates = jax.lax.cond(
        should_insert,
        lambda: top_rates.at[min_idx].set(new_rate),
        lambda: top_rates,
    )
    new_top_sel = jax.lax.cond(
        should_insert,
        lambda: top_configs.selected.at[min_idx].set(new_config.selected),
        lambda: top_configs.selected,
    )
    new_top_tp = jax.lax.cond(
        should_insert,
        lambda: top_configs.tx_power.at[min_idx].set(new_config.tx_power),
        lambda: top_configs.tx_power,
    )
    new_top_mcs = jax.lax.cond(
        should_insert,
        lambda: top_configs.mcs.at[min_idx].set(new_config.mcs),
        lambda: top_configs.mcs,
    )
    return new_top_rates, SAConfig(new_top_sel, new_top_tp, new_top_mcs)


def calibrate_T0(
    scenario,
    info:              ScenarioInfo,
    neighbor_fn:       Callable,
    seed:              int   = 42,
    n_steps:           int   = 200,
    target_acceptance: float = 0.80,
) -> float:
    """Calibrate T_0 so that the first n_steps have ~target_acceptance mean acceptance.

    Runs n_steps with T=∞ (accept all) to collect an unbiased delta distribution,
    then binary-searches for the T that achieves target_acceptance.
    """
    to_arrays      = make_config_to_arrays(info)
    random_config  = make_random_config(info)
    valid_mask_jax = jnp.array(info.valid_mask, dtype=jnp.int32)
    T_large        = jnp.float32(1e10)

    key          = jax.random.PRNGKey(seed)
    key, cfg_key = jax.random.split(key)
    current      = random_config(cfg_key)

    key, sim_key    = jax.random.split(key)
    tx, tx_p, mcs   = to_arrays(current)
    current_rate    = float(scenario(sim_key, tx, tx_p, mcs, return_internals=True)[0])

    deltas: list[float] = []

    for _ in range(n_steps):
        key, nbr_key, sim_key = jax.random.split(key, 3)
        candidate      = neighbor_fn(current, nbr_key, T_large, T_large, valid_mask_jax)
        tx, tx_p, mcs  = to_arrays(candidate)
        candidate_rate = float(scenario(sim_key, tx, tx_p, mcs, return_internals=True)[0])
        deltas.append(candidate_rate - current_rate)
        current      = candidate
        current_rate = candidate_rate

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


def make_sa_runner(
    scenario,
    info:        ScenarioInfo,
    neighbor_fn: Callable,
    T_0:         float,
    T_decay:     float,
    n_steps:     int,
    top_n:       int,
) -> Callable:
    """Build and return a JIT-compiled SA runner for the given scenario.

    The returned fn has signature:
        run(key, initial_config, initial_rate) -> SAResult

    n_steps and top_n are baked in at construction time (compile-time constants).
    """
    to_arrays  = make_config_to_arrays(info)
    valid_mask = jnp.array(info.valid_mask, dtype=jnp.int32)
    n_aps      = info.n_aps
    max_stas   = info.max_stas
    T_0_f      = jnp.float32(T_0)
    T_decay_f  = jnp.float32(T_decay)

    def _evaluate(config: SAConfig, key: jax.Array) -> jax.Array:
        tx, tx_p, mcs = to_arrays(config)
        rate, *_      = scenario(key, tx, tx_p, mcs, return_internals=True)
        return rate.astype(jnp.float32)

    def _step(state: SAState, step_idx: jax.Array) -> tuple[SAState, jax.Array]:
        T   = T_0_f * (T_decay_f ** step_idx.astype(jnp.float32))
        key, nbr_key, sim_key, acc_key = jax.random.split(state.key, 4)

        candidate      = neighbor_fn(state.config, nbr_key, T, T_0_f, valid_mask)
        candidate_rate = _evaluate(candidate, sim_key)

        delta      = candidate_rate - state.current_rate
        log_accept = jnp.minimum(jnp.float32(0.0), delta / jnp.maximum(T, jnp.float32(1e-300)))
        accept     = jnp.log(jax.random.uniform(acc_key)) < log_accept

        new_config = jax.lax.cond(accept, lambda: candidate,    lambda: state.config)
        new_rate   = jnp.where(accept,    candidate_rate,       state.current_rate)

        new_best_config, new_best_rate = jax.lax.cond(
            new_rate > state.best_rate,
            lambda: (new_config, new_rate),
            lambda: (state.best_config, state.best_rate),
        )

        new_top_rates, new_top_configs = _update_top_n(
            state.top_rates, state.top_configs, new_rate, new_config,
        )

        new_state = SAState(
            config       = new_config,
            current_rate = new_rate,
            best_config  = new_best_config,
            best_rate    = new_best_rate,
            top_rates    = new_top_rates,
            top_configs  = new_top_configs,
            key          = key,
        )
        return new_state, new_rate

    @jax.jit
    def run(
        key:            jax.Array,
        initial_config: SAConfig,
        initial_rate:   jax.Array,
    ) -> tuple[SAState, jax.Array]:
        init_state = SAState(
            config       = initial_config,
            current_rate = initial_rate,
            best_config  = initial_config,
            best_rate    = initial_rate,
            top_rates    = jnp.full((top_n,), -jnp.inf, dtype=jnp.float32),
            top_configs  = SAConfig(
                selected = jnp.zeros((top_n, n_aps, max_stas), dtype=jnp.int32),
                tx_power = jnp.zeros((top_n, n_aps, max_stas), dtype=jnp.int32),
                mcs      = jnp.zeros((top_n, n_aps, max_stas), dtype=jnp.int32),
            ),
            key          = key,
        )
        return jax.lax.scan(_step, init_state, jnp.arange(n_steps, dtype=jnp.int32))

    def simulated_annealing(
        key:            jax.Array,
        initial_config: SAConfig,
        initial_rate:   jax.Array,
    ) -> SAResult:
        final_state, history = run(key, initial_config, initial_rate)

        order   = jnp.argsort(-final_state.top_rates)
        s_rates = final_state.top_rates[order]
        s_sel   = final_state.top_configs.selected[order]
        s_tp    = final_state.top_configs.tx_power[order]
        s_mcs   = final_state.top_configs.mcs[order]

        top_list = [
            (float(s_rates[i]), SAConfig(s_sel[i], s_tp[i], s_mcs[i]))
            for i in range(top_n)
            if float(s_rates[i]) > -jnp.inf
        ]

        return SAResult(
            best_config = final_state.best_config,
            best_rate   = float(final_state.best_rate),
            top_configs = top_list,
            history     = [float(r) for r in np.array(history)],
            T_0         = T_0,
        )

    return simulated_annealing
