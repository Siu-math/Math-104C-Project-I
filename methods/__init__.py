"""Numerical methods for first-order initial value problems."""

from .adams_bashforth import solve as adams_bashforth_solve
from .adams_moulton import solve as adams_moulton_solve
from .euler import solve as euler_solve
from .heun import solve as heun_solve
from .midpoint import solve as midpoint_solve
from .modified_euler import solve as modified_euler_solve
from .predictor_corrector import solve as predictor_corrector_solve
from .rk4 import solve as rk4_solve
from .taylor import solve as taylor_solve

METHODS = [
    ("Euler", euler_solve),
    ("Taylor Order 2", taylor_solve),
    ("Midpoint", midpoint_solve),
    ("Modified Euler", modified_euler_solve),
    ("Heun Order 3", heun_solve),
    ("RK4", rk4_solve),
    ("Adams-Bashforth 4", adams_bashforth_solve),
    ("Adams-Moulton 3", adams_moulton_solve),
    ("Predictor-Corrector", predictor_corrector_solve),
]
