from collections.abc import Callable

from mapc_mh.methods.throughput.sa   import run as run_t_sa
from mapc_mh.methods.throughput.rrhc import run as run_t_rrhc
from mapc_mh.methods.throughput.tabu import run as run_t_tabu
from mapc_mh.methods.fairness.sa     import run as run_f_sa
from mapc_mh.methods.core            import Result
from mapc_mh.methods.fairness.core   import FResult

# T-Optimal family
T_METHODS: dict[str, Callable[..., Result]] = {
    'sa':   run_t_sa,
    'rrhc': run_t_rrhc,
    'tabu': run_t_tabu,
}

T_METHOD_LABELS: dict[str, str] = {
    'sa':   'SA',
    'rrhc': 'RRHC',
    'tabu': 'Tabu',
}

# F-Optimal family
F_METHODS: dict[str, Callable[..., FResult]] = {
    'f_sa': run_f_sa,
}

F_METHOD_LABELS: dict[str, str] = {
    'f_sa': 'F-SA',
}

BASELINE_LABELS: dict[str, str] = {
    'h_mab':     'H-MAB',
    'dcf':       'DCF',
    't_optimal': 'T-Optimal',
}

ALL_LABELS: dict[str, str] = {**T_METHOD_LABELS, **F_METHOD_LABELS, **BASELINE_LABELS}

__all__ = [
    'run_t_sa', 'run_t_rrhc', 'run_t_tabu', 'run_f_sa',
    'T_METHODS', 'T_METHOD_LABELS',
    'F_METHODS', 'F_METHOD_LABELS',
    'BASELINE_LABELS', 'ALL_LABELS',
    'Result', 'FResult',
]
