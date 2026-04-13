"""Pluggable weight-update strategies for F-Optimal evaluation.

The default `identity_strategy` lets the metaheuristic control logits directly via
the perturb_weights and add/remove moves. For ablation, swap in `lp_strategy` to
freeze the configs and let an LP solver find the optimal weights given the per-config
per-station throughput matrix.

Note: LP strategies cannot run inside `jax.jit`. If an LP strategy is used, the
`jax.lax.scan` main loop in the SA (or other algorithm) must be replaced with a
Python-level for loop. This is acceptable for ablation runs; not for the main path.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from mapc_mh.methods.fairness.core import FSolution

# WeightStrategy: (FSolution, sub_sta_rates: (max_configs, n_stas)) -> logits (max_configs,)
WeightStrategy = Callable[[FSolution, jax.Array], jax.Array]


def identity_strategy(sol: FSolution, sub_sta_rates: jax.Array) -> jax.Array:
    """Return the solution's own logits unchanged (metaheuristic controls weights)."""
    return sol.logits


def lp_strategy(sol: FSolution, sub_sta_rates: jax.Array) -> jax.Array:
    """Solve a max-min LP over the fixed sub-configs to find optimal mixing weights.

    Not yet implemented. When implemented:
    - sub_sta_rates shape: (max_configs, n_stas)
    - Solve: max t  s.t.  sum_m w_m * r_{m,s} >= t  for all s,  sum_m w_m = 1,  w_m >= 0
    - Convert optimal w -> logits via log(w + eps) and shift so softmax recovers w
    - Cannot be JIT-compiled (PuLP is Python-side); use a Python for loop instead of scan.
    """
    raise NotImplementedError(
        "lp_strategy: implement with PuLP or scipy.linprog; "
        "replace jax.lax.scan with a Python for loop in the calling algorithm."
    )
