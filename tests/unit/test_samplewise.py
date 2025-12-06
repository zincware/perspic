"""Tests for shared SamplewiseCalculator base class functionality."""

import warnings

import pytest
import torch
import torch.nn as nn

from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus
from perspic.utils import BatchStatSnapshot


class BatchNormMLP(nn.Module):
    """MLP with BatchNorm for testing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleMLP(nn.Module):
    """MLP without BatchNorm for testing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        return self.layers(x)


class TestBatchNormTrainingWarning:
    """Test that both calculators warn when BatchNorm is in training mode."""

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_warning_when_batchnorm_in_training_mode_network(self, calculator):
        """Verify warning is raised for network gradient computation."""
        torch.manual_seed(42)
        model = BatchNormMLP()
        model.train()
        X = torch.randn(8, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # The computation will fail with BatchNorm in training mode,
            # but the warning should be issued before the failure
            try:
                calculator._compute_per_sample_gradient_norm_network(model, X)
            except (ValueError, RuntimeError):
                # Expected - BatchNorm in training mode fails with vmap/sample-wise ops
                pass

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 1
            assert issubclass(bn_warnings[0].category, UserWarning)
            assert "BatchStatSnapshot" in str(bn_warnings[0].message)

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_warning_when_batchnorm_in_training_mode_loss(self, calculator):
        """Verify warning is raised for loss gradient computation."""
        torch.manual_seed(42)
        model = BatchNormMLP()
        model.train()
        X = torch.randn(8, 10)
        y = torch.randint(0, 2, (8,))
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculator._compute_per_sample_gradient_norm_loss(model, loss_fn, X, y)

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 1
            assert issubclass(bn_warnings[0].category, UserWarning)
            assert "BatchStatSnapshot" in str(bn_warnings[0].message)

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_no_warning_with_batch_stat_snapshot(self, calculator):
        """Verify no warning when BatchStatSnapshot is used."""
        torch.manual_seed(42)
        model = BatchNormMLP()
        X = torch.randn(8, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with BatchStatSnapshot(model, X):
                calculator._compute_per_sample_gradient_norm_network(model, X)

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 0

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_no_warning_without_batchnorm(self, calculator):
        """Verify no warning when model has no BatchNorm layers."""
        torch.manual_seed(42)
        model = SimpleMLP()
        model.train()
        X = torch.randn(8, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculator._compute_per_sample_gradient_norm_network(model, X)

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 0

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_no_warning_when_batchnorm_in_eval_mode(self, calculator):
        """Verify no warning when BatchNorm is in eval mode (even without snapshot)."""
        torch.manual_seed(42)
        model = BatchNormMLP()
        model.eval()  # Explicitly set to eval mode
        X = torch.randn(8, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculator._compute_per_sample_gradient_norm_network(model, X)

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 0
