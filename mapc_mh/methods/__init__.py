from collections.abc import Callable

from mapc_mh.methods.throughput.sa   import run as run_t_sa
from mapc_mh.methods.throughput.rrhc import run as run_t_rrhc
from mapc_mh.methods.throughput.tabu import run as run_t_tabu
from mapc_mh.methods.fairness.sa     import run as run_f_sa
from mapc_mh.methods.fairness.vns    import run as run_f_vns
from mapc_mh.methods.fairness.cg     import run as run_f_cg
from mapc_mh.methods.core            import Result
from mapc_mh.methods.fairness.core   import FResult

# T-Optimal family
T_METHODS: dict[str, Callable[..., Result]] = {
    't_sa':   run_t_sa,
    't_rrhc': run_t_rrhc,
    't_tabu': run_t_tabu,
}

T_METHOD_LABELS: dict[str, str] = {
    't_sa':   'T-SA',
    't_rrhc': 'T-RRHC',
    't_tabu': 'T-Tabu',
}

# F-Optimal family
F_METHODS: dict[str, Callable[..., FResult]] = {
    'f_sa':  run_f_sa,
    'f_vns': run_f_vns,
    'f_cg':  run_f_cg,
}

F_METHOD_LABELS: dict[str, str] = {
    'f_sa':  'F-SA',
    'f_vns': 'F-VNS',
    'f_cg':  'F-CG',
}

BASELINE_LABELS: dict[str, str] = {
    'h_mab':     'H-MAB',
    'dcf':       'DCF',
    't_optimal': 'T-Optimal',
    'f_optimal': 'F-Optimal',
}

ALL_LABELS: dict[str, str] = {**T_METHOD_LABELS, **F_METHOD_LABELS, **BASELINE_LABELS}

__all__ = [
    'run_t_sa', 'run_t_rrhc', 'run_t_tabu', 'run_f_sa', 'run_f_vns', 'run_f_cg',
    'T_METHODS', 'T_METHOD_LABELS',
    'F_METHODS', 'F_METHOD_LABELS',
    'BASELINE_LABELS', 'ALL_LABELS',
    'Result', 'FResult',
]
