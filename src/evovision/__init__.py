"""evovision: multi-objective evolutionary NAS for efficient vision."""

from evovision import search_space
from evovision.evolve import evolve
from evovision.models import ConvNet, build_model
from evovision.problem import REFERENCE_POINT, make_problem
from evovision.search_space import DIM, N_STAGES, bounds, flops, params, to_config
from evovision.training import train_and_eval

__version__ = "0.1.0"

__all__ = [
    "evolve",
    "make_problem",
    "REFERENCE_POINT",
    "train_and_eval",
    "build_model",
    "ConvNet",
    "to_config",
    "bounds",
    "flops",
    "params",
    "DIM",
    "N_STAGES",
    "search_space",
]
