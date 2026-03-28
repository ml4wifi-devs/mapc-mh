"""
Version-specific SA runners.

Each runner:
1. Builds ScenarioInfo from the scenario
2. Calibrates T_0 (or uses the provided one)
3. Creates a JIT-compiled SA runner via make_sa_runner
4. Generates an initial config, evaluates it, then runs SA
5. Returns SAResult
"""
from __future__ import annotations

import mapc_sa._env  # noqa: F401

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from mapc_sa.annealing import SAResult, calibrate_T0, make_sa_runner
from mapc_sa.config import make_config_to_arrays, make_random_config, make_scenario_info
from mapc_sa.neighbor import neighbor_v1, neighbor_v2, neighbor_v3, neighbor_v4


def _run(
    scenario,
    neighbor_fn,
    seed:    int,
    T_decay: float,
    T_0:     float | None,
    n_steps: int,
    top_n:   int,
) -> SAResult:
    info           = make_scenario_info(scenario)
    to_arrays      = make_config_to_arrays(info)
    random_config  = make_random_config(info)
    valid_mask_jax = jnp.array(info.valid_mask, dtype=jnp.int32)

    if T_0 is None:
        T_0 = calibrate_T0(scenario, info, neighbor_fn, seed=seed)

    sa_run = make_sa_runner(scenario, info, neighbor_fn, T_0, T_decay, n_steps, top_n)

    key              = jax.random.PRNGKey(seed)
    key, cfg_key, sim_key, run_key = jax.random.split(key, 4)

    initial_config   = random_config(cfg_key)
    tx, tx_p, mcs    = to_arrays(initial_config)
    initial_rate     = scenario(sim_key, tx, tx_p, mcs, return_internals=True)[0].astype(jnp.float32)

    result = sa_run(run_key, initial_config, initial_rate)
    return SAResult(
        best_config = result.best_config,
        best_rate   = result.best_rate,
        top_configs = result.top_configs,
        history     = result.history,
        T_0         = result.T_0,
        info        = info,
    )


def run_sa_v1(
    scenario,
    seed:    int   = 42,
    T_decay: float = 0.999,
    T_0:     float | None = None,
    n_steps: int   = 2000,
    top_n:   int   = 10,
) -> SAResult:
    """Version 1: pure SA — random draw from all valid values."""
    nbr = partial(neighbor_v1)
    return _run(scenario, nbr, seed, T_decay, T_0, n_steps, top_n)


def run_sa_v2(
    scenario,
    seed:    int   = 42,
    T_decay: float = 0.999,
    T_0:     float | None = None,
    n_steps: int   = 2000,
    top_n:   int   = 10,
) -> SAResult:
    """Version 2: ordinal-aware SA — tx_power/mcs change by ±1 only."""
    nbr = partial(neighbor_v2)
    return _run(scenario, nbr, seed, T_decay, T_0, n_steps, top_n)


def run_sa_v3(
    scenario,
    seed:          int   = 42,
    T_decay:       float = 0.999,
    T_0:           float | None = None,
    max_jump_mcs:  int   = 7,
    max_jump_power: int  = 3,
    n_steps:       int   = 2000,
    top_n:         int   = 10,
) -> SAResult:
    """Version 3: temperature-dependent jump sizes for ordinal params."""
    nbr = partial(neighbor_v3, max_jump_mcs=max_jump_mcs, max_jump_power=max_jump_power)
    return _run(scenario, nbr, seed, T_decay, T_0, n_steps, top_n)


def run_sa_v4(
    scenario,
    seed:               int   = 42,
    T_decay:            float = 0.999,
    T_0:                float | None = None,
    max_jump_mcs:       int   = 7,
    max_jump_power:     int   = 3,
    concurrency_target: float = 4.0,
    n_steps:            int   = 2000,
    top_n:              int   = 10,
) -> SAResult:
    """Version 4: concurrency-controlled SA with temperature-dependent jumps."""
    nbr = partial(neighbor_v4,
                  max_jump_mcs=max_jump_mcs,
                  max_jump_power=max_jump_power,
                  concurrency_target=concurrency_target)
    return _run(scenario, nbr, seed, T_decay, T_0, n_steps, top_n)


VERSION_RUNNERS = {1: run_sa_v1, 2: run_sa_v2, 3: run_sa_v3, 4: run_sa_v4}
