"""
Perspic: A tool to study neural network training dynamics.
"""
from perspic.analyzer import analyzer
from perspic.calculator.linearizer import Linearizer
from perspic.calculator.samplewise import SamplewiseCalculatorFunctorch

__all__ = [
    analyzer.__name__,
    Linearizer.__name__,
    SamplewiseCalculatorFunctorch.__name__,
]