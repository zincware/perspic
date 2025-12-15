"""
Perspic: A tool to study neural network training dynamics.
"""

from perspic.analyzer import analyzer
from perspic.calculator.linearizer import Linearizer
from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus
from perspic.logger import (
    LogarithmicWindowSchedule,
    logarithmic_windows,
)
from perspic.utils import MultiEpochsDataLoader

__all__ = [
    analyzer.__name__,
    Linearizer.__name__,
    SamplewiseCalculatorFunctorch.__name__,
    SamplewiseCalculatorOpacus.__name__,
    LogarithmicWindowSchedule.__name__,
    logarithmic_windows.__name__,
    MultiEpochsDataLoader.__name__,
]
