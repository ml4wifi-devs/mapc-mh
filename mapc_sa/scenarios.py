import mapc_sa.env  # noqa: F401

from mapc_research.envs.scenario_impl import residential_scenario

SCENARIO_CONFIGS = [
    (2, 1), (3, 1),
    (2, 2), (3, 2),
    (3, 3), (4, 3),
    (4, 4), (5, 4),
    (5, 5),
]

N_SEEDS = 5


def build_scenarios(n_seeds: int = N_SEEDS) -> list:
    """Build scenario list with n_seeds realizations per config.

    Seeds are always 0, 1, ..., n_seeds-1, so build_scenarios(3) produces
    the same first-3 scenarios per config as build_scenarios(5).
    """
    return [
        residential_scenario(
            seed=s, n_steps=2000,
            x_apartments=x, y_apartments=y,
            n_sta_per_ap=4, size=10.0,
            channel_width=80,
        )
        for x, y in SCENARIO_CONFIGS
        for s in range(n_seeds)
    ]


SCENARIOS = build_scenarios()
