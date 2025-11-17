"""
Perspic: A tool to study neural network training dynamics.
"""
from perspic.calculator.coupling import CouplingCalculator
from perspic.calculator.linearizer import Linearizer
from perspic.calculator.samplewise import SamplewiseCalculatorFunctorch

__all__ = [
    CouplingCalculator.__name__,
    Linearizer.__name__,
    SamplewiseCalculatorFunctorch.__name__,
]