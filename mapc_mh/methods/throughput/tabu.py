"""Tabu Search.

Config hash: weighted sum of flattened arrays using the first K prime numbers.
Tabu list: fixed-size ring buffer of config hashes.
Aspiration: override tabu if candidate improves upon the global best.
"""
from __future__ import annotations

import mapc_mh.env  # noqa: F401

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mapc_mh.config import NetworkConfig, make_config_to_arrays
from mapc_mh.methods.core import _build_result, _setup, _update_top_n, neighbor
from mapc_mh.methods.core import Result


def _first_n_primes(n: int) -> list[int]:
    primes, candidate = [], 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


class _TabuState(NamedTuple):
    config:       NetworkConfig
    current_rate: jax.Array
    best_config:  NetworkConfig
    best_rate:    jax.Array
    top_rates:    jax.Array
    top_configs:  NetworkConfig
    tabu_hashes:  jax.Array
    tabu_ptr:     jax.Array
    key:          jax.Array


def _make_runner(scenario, info, n_steps: int, top_n: int, tabu_size: int, n_candidates: int):
    to_arrays  = make_config_to_arrays(info)
    valid_mask = jnp.array(info.valid_mask, dtype=jnp.int32)
    n_aps      = info.n_aps
    max_stas   = info.max_stas
    primes     = jnp.array(_first_n_primes(n_aps * max_stas * 3), dtype=jnp.int32)

    def _hash(config: NetworkConfig) -> jax.Array:
        flat = jnp.concatenate([config.selected.ravel(), config.tx_power.ravel(), config.mcs.ravel()])
        return jnp.sum(flat * primes) & 0x7FFFFFFF  # always non-negative; -1 is safe sentinel

    def _evaluate(config: NetworkConfig, key: jax.Array) -> jax.Array:
        tx, tx_p, mcs = to_arrays(config)
        return scenario(key, tx, tx_p, mcs, return_internals=True)[0].astype(jnp.float32)

    def _step(state: _TabuState, _):
        key, *cand_keys = jax.random.split(state.key, n_candidates + 1)
        cand_keys = jnp.stack(cand_keys)

        def gen_one(k):
            k, nbr_key, sim_key = jax.random.split(k, 3)
            cand = neighbor(state.config, nbr_key, valid_mask)
            rate = _evaluate(cand, sim_key)
            return cand.selected, cand.tx_power, cand.mcs, rate, _hash(cand)

        c_sel, c_tp, c_mcs, c_rates, c_hashes = jax.vmap(gen_one)(cand_keys)

        is_tabu      = jax.vmap(lambda h: jnp.any(state.tabu_hashes == h))(c_hashes)
        aspirated    = c_rates > state.best_rate
        blocked      = is_tabu & ~aspirated
        scores       = c_rates + jnp.where(blocked, jnp.float32(-1e10), jnp.float32(0.0))
        best_idx     = jnp.argmax(scores)

        new_config   = NetworkConfig(c_sel[best_idx], c_tp[best_idx], c_mcs[best_idx])
        new_rate     = c_rates[best_idx]

        old_hash     = _hash(state.config)
        new_hashes   = state.tabu_hashes.at[state.tabu_ptr].set(old_hash)
        new_ptr      = (state.tabu_ptr + 1) % tabu_size

        new_best_config, new_best_rate = jax.lax.cond(
            new_rate > state.best_rate,
            lambda: (new_config, new_rate),
            lambda: (state.best_config, state.best_rate),
        )
        new_top_rates, new_top_configs = _update_top_n(
            state.top_rates, state.top_configs, new_rate, new_config,
        )
        return _TabuState(
            config       = new_config,
            current_rate = new_rate,
            best_config  = new_best_config,
            best_rate    = new_best_rate,
            top_rates    = new_top_rates,
            top_configs  = new_top_configs,
            tabu_hashes  = new_hashes,
            tabu_ptr     = new_ptr,
            key          = key,
        ), (new_rate, new_best_rate)

    @jax.jit
    def _run(key, initial_config, initial_rate):
        init_state = _TabuState(
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
            tabu_hashes  = jnp.full((tabu_size,), jnp.int32(-1)),
            tabu_ptr     = jnp.int32(0),
            key          = key,
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
    seed:         int = 42,
    n_steps:      int = 2000,
    top_n:        int = 10,
    tabu_size:    int = 20,
    n_candidates: int = 10,
) -> Result:
    """Run Tabu Search on *scenario*."""
    info, run_key, initial_config, initial_rate = _setup(scenario, seed)
    runner = _make_runner(scenario, info, n_steps, top_n, tabu_size, n_candidates)
    return runner(run_key, initial_config, initial_rate)
