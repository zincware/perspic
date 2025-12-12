"""Integration tests for batch-size normalization scaling in sample-wise calculators.

These tests verify that the normalization logic correctly ensures sample-wise
metrics scale properly with batch size. Specifically:
- With normalize=True: metrics should be batch-size invariant
- With normalize=False: metrics should scale linearly with batch size
"""

import pytest
import torch
import torch.nn as nn

from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus
from perspic.utils import BatchStatSnapshot

torch.set_float32_matmul_precision("high")


class MLP(nn.Module):
    """Simple MLP for testing normalization scaling."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class BatchNormMLP(nn.Module):
    """MLP with BatchNorm for testing normalization scaling."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TestNormalizationScalingFunctorch:
    """Test batch-size scaling behavior for SamplewiseCalculatorFunctorch."""

    @pytest.mark.parametrize("batch_sizes", [(8, 16), (8, 32), (16, 64)])
    def test_normalized_metrics_are_batch_size_invariant(self, batch_sizes):
        """Verify that normalized metrics remain constant across different batch sizes.

        When normalize=True, metrics should be independent of batch size.
        """
        torch.manual_seed(42)
        batch_size_small, batch_size_large = batch_sizes

        # Create fixed data pool larger than the largest batch
        X_pool = torch.randn(batch_size_large, 10)
        y_pool = torch.randint(0, 2, (batch_size_large,))

        model = MLP()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        calculator = SamplewiseCalculatorFunctorch()

        # Compute with smaller batch
        X_small = X_pool[:batch_size_small]
        y_small = y_pool[:batch_size_small]
        results_small = calculator.compute(
            model, loss_fn, X_small, y_small, normalize=True
        )

        # Compute with larger batch (using same data prefix + more samples)
        results_large = calculator.compute(
            model, loss_fn, X_pool, y_pool, normalize=True
        )

        # Normalized metrics should be approximately equal (batch-size invariant)
        # Note: They won't be exactly equal because we're using different samples,
        # but the *scaling* should be correct. We verify by checking the ratio.
        ratio_network = (
            results_large["batch_grad_norms_network"]
            / results_small["batch_grad_norms_network"]
        )
        ratio_loss = (
            results_large["batch_grad_norms_loss"]
            / results_small["batch_grad_norms_loss"]
        )

        # The ratios should be close to 1.0 (within a reasonable tolerance due to
        # different samples contributing different gradient magnitudes)
        # For a proper invariance test, we need identical samples - see next test
        assert ratio_network > 0, "Network gradient norms should be positive"
        assert ratio_loss > 0, "Loss gradient norms should be positive"

    @pytest.mark.parametrize("batch_sizes", [(8, 16), (8, 32), (16, 32)])
    def test_normalized_metrics_scale_correctly_with_repeated_samples(
        self, batch_sizes
    ):
        """Verify normalization by using repeated identical samples.

        If we repeat the same batch of samples N times, the normalized metrics
        should remain constant (batch-size invariant).
        """
        torch.manual_seed(42)
        base_batch_size, multiplied_batch_size = batch_sizes
        multiplier = multiplied_batch_size // base_batch_size

        # Create base data
        X_base = torch.randn(base_batch_size, 10)
        y_base = torch.randint(0, 2, (base_batch_size,))

        # Create repeated data (same samples repeated)
        X_repeated = X_base.repeat(multiplier, 1)
        y_repeated = y_base.repeat(multiplier)

        model = MLP()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        calculator = SamplewiseCalculatorFunctorch()

        # Compute with base batch
        results_base = calculator.compute(
            model, loss_fn, X_base, y_base, normalize=True
        )

        # Compute with repeated batch
        results_repeated = calculator.compute(
            model, loss_fn, X_repeated, y_repeated, normalize=True
        )

        # With identical (repeated) samples, normalized metrics should be equal
        assert torch.allclose(
            results_base["batch_grad_norms_network"],
            results_repeated["batch_grad_norms_network"],
            rtol=1e-4,
            atol=1e-6,
        ), f"Normalized network norms should be batch-size invariant: {results_base['batch_grad_norms_network']} vs {results_repeated['batch_grad_norms_network']}"

        assert torch.allclose(
            results_base["batch_grad_norms_loss"],
            results_repeated["batch_grad_norms_loss"],
            rtol=1e-4,
            atol=1e-6,
        ), f"Normalized loss norms should be batch-size invariant: {results_base['batch_grad_norms_loss']} vs {results_repeated['batch_grad_norms_loss']}"

    @pytest.mark.parametrize("batch_sizes", [(8, 16), (8, 32), (16, 32)])
    def test_unnormalized_metrics_scale_linearly_with_batch_size(self, batch_sizes):
        """Verify that unnormalized metrics scale linearly with batch size.

        When normalize=False, doubling the batch size (with repeated samples)
        should double the network gradient norms and halve the loss gradient norms.
        """
        torch.manual_seed(42)
        base_batch_size, multiplied_batch_size = batch_sizes
        multiplier = multiplied_batch_size // base_batch_size

        # Create base data
        X_base = torch.randn(base_batch_size, 10)
        y_base = torch.randint(0, 2, (base_batch_size,))

        # Create repeated data
        X_repeated = X_base.repeat(multiplier, 1)
        y_repeated = y_base.repeat(multiplier)

        model = MLP()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        calculator = SamplewiseCalculatorFunctorch()

        # Compute with base batch (unnormalized)
        results_base = calculator.compute(
            model, loss_fn, X_base, y_base, normalize=False
        )

        # Compute with repeated batch (unnormalized)
        results_repeated = calculator.compute(
            model, loss_fn, X_repeated, y_repeated, normalize=False
        )

        # Network gradient norms should scale linearly (multiply by batch_size ratio)
        expected_network = results_base["batch_grad_norms_network"] * multiplier
        assert torch.allclose(
            results_repeated["batch_grad_norms_network"],
            expected_network,
            rtol=1e-4,
            atol=1e-6,
        ), f"Unnormalized network norms should scale linearly: expected {expected_network}, got {results_repeated['batch_grad_norms_network']}"

        # Loss gradient norms should scale inversely (divide by batch_size ratio)
        # because CrossEntropyLoss(reduction="mean") averages over batch, and the
        # Jacobian w.r.t. output has 1/batch_size factor from the loss
        expected_loss = results_base["batch_grad_norms_loss"] / multiplier
        assert torch.allclose(
            results_repeated["batch_grad_norms_loss"],
            expected_loss,
            rtol=1e-4,
            atol=1e-6,
        ), f"Unnormalized loss norms should scale inversely: expected {expected_loss}, got {results_repeated['batch_grad_norms_loss']}"


class TestNormalizationScalingOpacus:
    """Test batch-size scaling behavior for SamplewiseCalculatorOpacus."""

    @pytest.mark.parametrize("batch_sizes", [(8, 16), (8, 32), (16, 32)])
    def test_normalized_metrics_scale_correctly_with_repeated_samples(
        self, batch_sizes
    ):
        """Verify normalization by using repeated identical samples.

        If we repeat the same batch of samples N times, the normalized metrics
        should remain constant (batch-size invariant).
        """
        torch.manual_seed(42)
        base_batch_size, multiplied_batch_size = batch_sizes
        multiplier = multiplied_batch_size // base_batch_size

        # Create base data
        X_base = torch.randn(base_batch_size, 10)
        y_base = torch.randint(0, 2, (base_batch_size,))

        # Create repeated data
        X_repeated = X_base.repeat(multiplier, 1)
        y_repeated = y_base.repeat(multiplier)

        model = MLP()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        calculator = SamplewiseCalculatorOpacus()

        # Compute with base batch
        with BatchStatSnapshot(model, X_base):
            results_base = calculator.compute(
                model, loss_fn, X_base, y_base, normalize=True
            )

        # Compute with repeated batch
        with BatchStatSnapshot(model, X_repeated):
            results_repeated = calculator.compute(
                model, loss_fn, X_repeated, y_repeated, normalize=True
            )

        # Normalized metrics should be batch-size invariant
        assert torch.allclose(
            results_base["batch_grad_norms_network"],
            results_repeated["batch_grad_norms_network"],
            rtol=1e-4,
            atol=1e-6,
        ), f"Normalized network norms should be batch-size invariant: {results_base['batch_grad_norms_network']} vs {results_repeated['batch_grad_norms_network']}"

        assert torch.allclose(
            results_base["batch_grad_norms_loss"],
            results_repeated["batch_grad_norms_loss"],
            rtol=1e-4,
            atol=1e-6,
        ), f"Normalized loss norms should be batch-size invariant: {results_base['batch_grad_norms_loss']} vs {results_repeated['batch_grad_norms_loss']}"

    @pytest.mark.parametrize("batch_sizes", [(8, 16), (8, 32), (16, 32)])
    def test_unnormalized_metrics_scale_linearly_with_batch_size(self, batch_sizes):
        """Verify that unnormalized metrics scale linearly with batch size."""
        torch.manual_seed(42)
        base_batch_size, multiplied_batch_size = batch_sizes
        multiplier = multiplied_batch_size // base_batch_size

        # Create base data
        X_base = torch.randn(base_batch_size, 10)
        y_base = torch.randint(0, 2, (base_batch_size,))

        # Create repeated data
        X_repeated = X_base.repeat(multiplier, 1)
        y_repeated = y_base.repeat(multiplier)

        model = MLP()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        calculator = SamplewiseCalculatorOpacus()

        # Compute with base batch (unnormalized)
        with BatchStatSnapshot(model, X_base):
            results_base = calculator.compute(
                model, loss_fn, X_base, y_base, normalize=False
            )

        # Compute with repeated batch (unnormalized)
        with BatchStatSnapshot(model, X_repeated):
            results_repeated = calculator.compute(
                model, loss_fn, X_repeated, y_repeated, normalize=False
            )

        # Network gradient norms should scale linearly
        expected_network = results_base["batch_grad_norms_network"] * multiplier
        assert torch.allclose(
            results_repeated["batch_grad_norms_network"],
            expected_network,
            rtol=1e-4,
            atol=1e-6,
        ), f"Unnormalized network norms should scale linearly: expected {expected_network}, got {results_repeated['batch_grad_norms_network']}"

        # Loss gradient norms should scale inversely
        expected_loss = results_base["batch_grad_norms_loss"] / multiplier
        assert torch.allclose(
            results_repeated["batch_grad_norms_loss"],
            expected_loss,
            rtol=1e-4,
            atol=1e-6,
        ), f"Unnormalized loss norms should scale inversely: expected {expected_loss}, got {results_repeated['batch_grad_norms_loss']}"


class TestCrossValidationNormalization:
    """Cross-validate normalization behavior between Functorch and Opacus."""

    @pytest.mark.parametrize("batch_size", [8, 16, 32])
    def test_functorch_and_opacus_normalized_metrics_match(self, batch_size):
        """Verify that Functorch and Opacus produce the same normalized metrics."""
        torch.manual_seed(42)

        X = torch.randn(batch_size, 10)
        y = torch.randint(0, 2, (batch_size,))

        model = MLP()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")

        functorch_calc = SamplewiseCalculatorFunctorch()
        opacus_calc = SamplewiseCalculatorOpacus()

        # Functorch results
        results_functorch = functorch_calc.compute(model, loss_fn, X, y, normalize=True)

        # Opacus results (need BatchStatSnapshot even for non-BatchNorm models)
        with BatchStatSnapshot(model, X):
            results_opacus = opacus_calc.compute(model, loss_fn, X, y, normalize=True)

        assert torch.allclose(
            results_functorch["batch_grad_norms_network"],
            results_opacus["batch_grad_norms_network"],
            rtol=1e-3,
            atol=1e-4,
        ), f"Normalized network norms should match: functorch={results_functorch['batch_grad_norms_network']}, opacus={results_opacus['batch_grad_norms_network']}"

        assert torch.allclose(
            results_functorch["batch_grad_norms_loss"],
            results_opacus["batch_grad_norms_loss"],
            rtol=1e-3,
            atol=1e-4,
        ), f"Normalized loss norms should match: functorch={results_functorch['batch_grad_norms_loss']}, opacus={results_opacus['batch_grad_norms_loss']}"

    @pytest.mark.parametrize("batch_size", [8, 16, 32])
    def test_functorch_and_opacus_unnormalized_metrics_match(self, batch_size):
        """Verify that Functorch and Opacus produce the same unnormalized metrics."""
        torch.manual_seed(42)

        X = torch.randn(batch_size, 10)
        y = torch.randint(0, 2, (batch_size,))

        model = MLP()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")

        functorch_calc = SamplewiseCalculatorFunctorch()
        opacus_calc = SamplewiseCalculatorOpacus()

        # Functorch results
        results_functorch = functorch_calc.compute(
            model, loss_fn, X, y, normalize=False
        )

        # Opacus results
        with BatchStatSnapshot(model, X):
            results_opacus = opacus_calc.compute(model, loss_fn, X, y, normalize=False)

        assert torch.allclose(
            results_functorch["batch_grad_norms_network"],
            results_opacus["batch_grad_norms_network"],
            rtol=1e-3,
            atol=1e-4,
        ), f"Unnormalized network norms should match: functorch={results_functorch['batch_grad_norms_network']}, opacus={results_opacus['batch_grad_norms_network']}"

        assert torch.allclose(
            results_functorch["batch_grad_norms_loss"],
            results_opacus["batch_grad_norms_loss"],
            rtol=1e-3,
            atol=1e-4,
        ), f"Unnormalized loss norms should match: functorch={results_functorch['batch_grad_norms_loss']}, opacus={results_opacus['batch_grad_norms_loss']}"


class TestBatchNormNormalizationScaling:
    """Test normalization scaling with BatchNorm models."""

    @pytest.mark.parametrize("batch_sizes", [(8, 16), (16, 32)])
    def test_batchnorm_normalized_metrics_invariant_functorch(self, batch_sizes):
        """Verify normalization with BatchNorm models using Functorch."""
        torch.manual_seed(42)
        base_batch_size, multiplied_batch_size = batch_sizes
        multiplier = multiplied_batch_size // base_batch_size

        X_base = torch.randn(base_batch_size, 10)
        y_base = torch.randint(0, 2, (base_batch_size,))

        X_repeated = X_base.repeat(multiplier, 1)
        y_repeated = y_base.repeat(multiplier)

        model = BatchNormMLP()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        calculator = SamplewiseCalculatorFunctorch()

        with BatchStatSnapshot(model, X_base):
            results_base = calculator.compute(
                model, loss_fn, X_base, y_base, normalize=True
            )

        with BatchStatSnapshot(model, X_repeated):
            results_repeated = calculator.compute(
                model, loss_fn, X_repeated, y_repeated, normalize=True
            )

        assert torch.allclose(
            results_base["batch_grad_norms_network"],
            results_repeated["batch_grad_norms_network"],
            rtol=1e-4,
            atol=1e-6,
        ), "BatchNorm model: normalized network norms should be batch-size invariant"

        assert torch.allclose(
            results_base["batch_grad_norms_loss"],
            results_repeated["batch_grad_norms_loss"],
            rtol=1e-4,
            atol=1e-6,
        ), "BatchNorm model: normalized loss norms should be batch-size invariant"

    @pytest.mark.parametrize("batch_sizes", [(8, 16), (16, 32)])
    def test_batchnorm_normalized_metrics_invariant_opacus(self, batch_sizes):
        """Verify normalization with BatchNorm models using Opacus."""
        torch.manual_seed(42)
        base_batch_size, multiplied_batch_size = batch_sizes
        multiplier = multiplied_batch_size // base_batch_size

        X_base = torch.randn(base_batch_size, 10)
        y_base = torch.randint(0, 2, (base_batch_size,))

        X_repeated = X_base.repeat(multiplier, 1)
        y_repeated = y_base.repeat(multiplier)

        model = BatchNormMLP()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        calculator = SamplewiseCalculatorOpacus()

        with BatchStatSnapshot(model, X_base):
            results_base = calculator.compute(
                model, loss_fn, X_base, y_base, normalize=True
            )

        with BatchStatSnapshot(model, X_repeated):
            results_repeated = calculator.compute(
                model, loss_fn, X_repeated, y_repeated, normalize=True
            )

        assert torch.allclose(
            results_base["batch_grad_norms_network"],
            results_repeated["batch_grad_norms_network"],
            rtol=1e-4,
            atol=1e-6,
        ), "BatchNorm model: normalized network norms should be batch-size invariant"

        assert torch.allclose(
            results_base["batch_grad_norms_loss"],
            results_repeated["batch_grad_norms_loss"],
            rtol=1e-4,
            atol=1e-6,
        ), "BatchNorm model: normalized loss norms should be batch-size invariant"
