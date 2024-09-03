"""Top-level package for slsim."""

__author__ = """DESC/SLSC"""
__version__ = "0.1.0"

# Import specific classes or functions from modules to make them accessible directly from the package
from .lens import Lens  # Importing the Lens class from the lens module
from .lensed_population_base import (
    LensedPopulationBase,
)  # Importing the LensedPopulationBase class
