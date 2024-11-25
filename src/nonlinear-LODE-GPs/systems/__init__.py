"""dynamic systems
"""

from .systems import ODE_System, System1, ThreeTank, Bipendulum
from .inverted_pendulum import Inverted_Pendulum
from .nonlinear_threetank import Nonlinear_ThreeTank
from .linearize import get_equilibrium_equations, solve_for_equilibrium
__all__ = [
    "ODE_System",
    "System1", 
    "ThreeTank", 
    "Bipendulum",
    "Inverted_Pendulum",
    "Nonlinear_ThreeTank",
    "get_equilibrium_equations",
    "solve_for_equilibrium",
]