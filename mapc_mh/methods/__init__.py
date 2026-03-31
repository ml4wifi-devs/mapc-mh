from collections.abc import Callable

from mapc_mh.methods.sa   import run as run_sa
from mapc_mh.methods.rrhc import run as run_rrhc
from mapc_mh.methods.tabu import run as run_tabu
from mapc_mh.methods.core import Result

METHODS: dict[str, Callable[..., Result]] = {
    'sa':   run_sa,
    'rrhc': run_rrhc,
    'tabu': run_tabu,
}

METHOD_LABELS: dict[str, str] = {
    'sa':   'SA',
    'rrhc': 'RRHC',
    'tabu': 'Tabu',
}

BASELINE_LABELS: dict[str, str] = {
    'h_mab':     'H-MAB',
    'dcf':       'DCF',
    't_optimal': 'T-Optimal',
}

ALL_LABELS: dict[str, str] = {**METHOD_LABELS, **BASELINE_LABELS}

__all__ = ['run_sa', 'run_rrhc', 'run_tabu', 'METHODS', 'METHOD_LABELS', 'BASELINE_LABELS', 'ALL_LABELS']
