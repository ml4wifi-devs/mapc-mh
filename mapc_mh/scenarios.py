import hashlib
import pathlib

import mapc_mh.env  # noqa: F401

import cloudpickle
import lz4.frame
from mapc_research.envs.scenario_impl import residential_scenario

SCENARIO_CONFIGS = [
    (2, 1), (3, 1),
    (2, 2), (3, 2),
    (3, 3), (4, 3),
    (4, 4), (5, 4),
    (5, 5),
]

N_SEEDS = 5

_CACHE_DIR = pathlib.Path("/tmp/mapc_mh_cache")


def _scenario_cache_path(seed: int, x: int, y: int, **kwargs) -> pathlib.Path:
    """One file per scenario, keyed on every parameter."""
    params = dict(seed=seed, x_apartments=x, y_apartments=y, **kwargs)
    key = hashlib.sha256(repr(sorted(params.items())).encode()).hexdigest()
    # Human-readable prefix so the tmp dir is inspectable
    prefix = f"s{seed}_x{x}_y{y}"
    return _CACHE_DIR / f"scenario__{prefix}__{key[:12]}.pkl.lz4"


def _load_or_build(seed: int, x: int, y: int, **kwargs):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _scenario_cache_path(seed, x, y, **kwargs)
    if path.exists():
        with lz4.frame.open(path, "rb") as f:
            return cloudpickle.load(f)
    scenario = residential_scenario(seed=seed, x_apartments=x, y_apartments=y, **kwargs)
    with lz4.frame.open(path, "wb") as f:
        cloudpickle.dump(scenario, f)
    return scenario


def build_scenarios(
    n_seeds: int = N_SEEDS,
    n_steps: int = 2000,
    n_sta_per_ap: int = 4,
    size: float = 10.0,
    channel_width: int = 80,
) -> list:
    """Build scenario list with n_seeds realizations per config.

    Each scenario is cached individually in /tmp/mapc_mh_cache/ (cloudpickle +
    lz4). Cache key encodes every parameter of that specific scenario, so:
    - changing n_seeds only builds the new seeds
    - changing any kwarg misses cache only for affected scenarios
    - different kwarg combinations coexist safely in the cache dir
    """
    return [
        _load_or_build(
            seed=s, x=x, y=y,
            n_steps=n_steps, n_sta_per_ap=n_sta_per_ap,
            size=size, channel_width=channel_width,
        )
        for x, y in SCENARIO_CONFIGS
        for s in range(n_seeds)
    ]
