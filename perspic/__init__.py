"""
Perspic: A tool to study neural network training dynamics.
"""

from perspic.analyzer import analyzer
from perspic.calculator.linearizer import (
    ApproximateLinearizer,
    BaseLinearizer,
    ExactLinearizer,
    Linearizer,
)
from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus
from perspic.logger import (
    LogarithmicWindowSchedule,
    logarithmic_windows,
)

__all__ = [
    analyzer.__name__,
    BaseLinearizer.__name__,
    ApproximateLinearizer.__name__,
    ExactLinearizer.__name__,
    Linearizer.__name__,  # Backwards compatibility alias
    SamplewiseCalculatorFunctorch.__name__,
    SamplewiseCalculatorOpacus.__name__,
    LogarithmicWindowSchedule.__name__,
    logarithmic_windows.__name__,
]
