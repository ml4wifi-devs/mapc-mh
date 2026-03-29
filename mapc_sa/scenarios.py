"""
Scenario definitions for tuning and evaluation.

Mirrors the definitions in mapc_surrogate/sim.py to ensure comparability.
"""
import mapc_sa._env  # noqa: F401

from mapc_research.envs.scenario_impl import random_scenario, residential_scenario

# Residential 2×2 apartment scenario used for final evaluation
RESIDENTIAL_SCENARIOS = [
    residential_scenario(
        seed=20, n_steps=2000,
        x_apartments=2, y_apartments=2,
        n_sta_per_ap=4, size=10.0,
        channel_width=80,
    ),
]

# Random scenarios with varying AP counts used for hyperparameter tuning
RANDOM_AP_COUNTS   = list(range(2, 17, 2))
N_RANDOM_SCENARIOS = 2

RANDOM_SCENARIOS = [
    random_scenario(
        seed=200 + n_ap * N_RANDOM_SCENARIOS + i,
        d_ap=75., d_sta=5.,
        n_ap=n_ap, n_sta_per_ap=4,
        n_steps=2000, channel_width=80,
        randomize=False,
    )
    for n_ap in RANDOM_AP_COUNTS
    for i in range(N_RANDOM_SCENARIOS)
]
