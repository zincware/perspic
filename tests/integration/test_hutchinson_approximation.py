"""Integration tests for Hutchinson's trace estimator approximation."""

import pytest
import torch
import torch.nn as nn

from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus
from perspic.utils import BatchStatSnapshot


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, input_dim=32, hidden_dim=64, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TestHutchinsonApproximationConvergence:
    """Test that Hutchinson's estimator converges to exact computation."""

    def test_approximation_converges_to_exact(self):
        """Verify that increasing n brings the approximation closer to exact."""
        torch.manual_seed(42)

        model = SimpleMLP(input_dim=32, hidden_dim=64, output_dim=10)
        model.eval()
        inputs = torch.randn(16, 32)

        with BatchStatSnapshot(model, inputs):
            # Compute exact value
            exact = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, inputs, reduce=True, approximate_with_n=None
                )
            )

            # Compute approximations with increasing n
            n_values = [1, 10, 100]
            errors = []

            for n in n_values:
                # Average over multiple runs for stability
                approx_values = []
                for _ in range(5):
                    approx = SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                        model, inputs, reduce=True, approximate_with_n=n
                    )
                    approx_values.append(approx.item())

                mean_approx = sum(approx_values) / len(approx_values)
                rel_error = abs(mean_approx - exact.item()) / exact.item()
                errors.append(rel_error)

            # Verify error decreases with larger n
            assert errors[1] < errors[0], "Error should decrease from n=1 to n=10"
            assert errors[2] < errors[1], "Error should decrease from n=10 to n=100"

    def test_approximation_is_unbiased(self):
        """Verify that the approximation is unbiased (mean converges to exact)."""
        torch.manual_seed(123)

        model = SimpleMLP(input_dim=16, hidden_dim=32, output_dim=5)
        model.eval()
        inputs = torch.randn(8, 16)

        with BatchStatSnapshot(model, inputs):
            # Compute exact value
            exact = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, inputs, reduce=True, approximate_with_n=None
                )
            )

            # Many runs with small n should average to exact
            n_runs = 50
            approx_sum = 0.0
            for _ in range(n_runs):
                approx = SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, inputs, reduce=True, approximate_with_n=5
                )
                approx_sum += approx.item()

            mean_approx = approx_sum / n_runs
            rel_error = abs(mean_approx - exact.item()) / exact.item()

            # With 50 runs, mean should be within 1% of exact
            assert (
                rel_error < 0.01
            ), f"Mean approximation error {rel_error:.2%} exceeds 1%"

    def test_per_sample_approximation_shape_preserved(self):
        """Verify that per-sample (reduce=False) preserves batch dimension."""
        torch.manual_seed(42)

        batch_size = 16
        model = SimpleMLP(input_dim=32, hidden_dim=64, output_dim=10)
        model.eval()
        inputs = torch.randn(batch_size, 32)

        with BatchStatSnapshot(model, inputs):
            # Exact per-sample norms
            exact = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, inputs, reduce=False, approximate_with_n=None
                )
            )

            # Approximate per-sample norms
            approx = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, inputs, reduce=False, approximate_with_n=10
                )
            )

            assert exact.shape == (batch_size,)
            assert approx.shape == (batch_size,)

    def test_high_n_matches_exact_closely(self):
        """With n equal to output_dim, approximation should be very close to exact."""
        torch.manual_seed(42)

        output_dim = 10
        model = SimpleMLP(input_dim=16, hidden_dim=32, output_dim=output_dim)
        model.eval()
        inputs = torch.randn(8, 16)

        with BatchStatSnapshot(model, inputs):
            exact = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, inputs, reduce=True, approximate_with_n=None
                )
            )

            # With n = output_dim * 10, should be very accurate
            approx = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, inputs, reduce=True, approximate_with_n=output_dim * 10
                )
            )
            approx_scaled = approx.item()

            rel_error = abs(approx_scaled - exact.item()) / exact.item()
            assert (
                rel_error < 0.05
            ), f"High-n approximation error {rel_error:.2%} exceeds 5%"
