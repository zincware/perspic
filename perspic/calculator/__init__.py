"""
Perspic: A tool to study neural network training dynamics.
"""

from perspic.calculator.coupling import CouplingCalculator
from perspic.calculator.linearizer import (
    ApproximateLinearizer,
    BaseLinearizer,
    ExactLinearizer,
)
from perspic.calculator.samplewise import SamplewiseCalculator
from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus

__all__ = [
    CouplingCalculator.__name__,
    BaseLinearizer.__name__,
    ApproximateLinearizer.__name__,
    ExactLinearizer.__name__,
    SamplewiseCalculator.__name__,
    SamplewiseCalculatorFunctorch.__name__,
    SamplewiseCalculatorOpacus.__name__,
]
