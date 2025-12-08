import math

import pytest
import torch
import torch.nn as nn

from perspic import Linearizer


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    return model


def complex_model():
    """Create a more complex model with BatchNorm."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


class TestLinearizer:
    """Tests for Linearizer linearizer functionality."""

    def test_simple_model_matches_analytical_gradient(self, simple_model):
        """Test that linearizer results match analytical first-order Taylor expansion."""
        torch.manual_seed(42)
        criterion = nn.MSELoss()
        x = torch.randn(8, 10)
        y = torch.randn(8, 5)

        # Save and compute gradient norm
        initial_state = {k: v.clone() for k, v in simple_model.state_dict().items()}
        simple_model.zero_grad()
        loss = criterion(simple_model(x), y)
        loss.backward()
        grad_norm_squared = sum(
            (p.grad**2).sum().item()
            for p in simple_model.parameters()
            if p.grad is not None
        )

        # Probe
        simple_model.load_state_dict(initial_state)
        simple_model.zero_grad()
        eta_array = [1e-3, 1e-4, 1e-5]
        results = Linearizer(eta_array).compute(
            model=simple_model,
            criterion=criterion,
            x=x,
            y=y,
        )

        # Verify Taylor approximation
        for eta, (loss_before, loss_after, delta_loss) in results.items():
            assert loss_after is not None
            assert delta_loss is not None
            actual_delta = delta_loss
            expected_delta = -eta * grad_norm_squared

            # Tolerance scales with eta: larger eta → larger tolerance
            tolerance = max(1e-5, abs(expected_delta) * 0.15)
            abs_error = abs(actual_delta - expected_delta)

            assert (
                abs_error < tolerance
            ), f"eta={eta}: error={abs_error:.8e}, tolerance={tolerance:.8e}"

    def test_complex_model_loss_delta_scales_linear(self):
        """Test that loss delta scales linearly with eta across multiple model initializations."""
        criterion = nn.MSELoss()
        x = torch.randn(8, 10)
        y = torch.randn(8, 5)

        etas = [1e-4, 1e-5, 1e-6]
        num_seeds = 50

        # Average deltas for each eta
        avg_deltas = {eta: 0.0 for eta in etas}

        for _seed in range(num_seeds):
            torch.manual_seed(_seed)
            model = complex_model()

            # Probe all etas with same model
            linearizer = Linearizer(etas)
            results = linearizer.compute(
                model=model,
                criterion=criterion,
                x=x,
                y=y,
            )

            # Accumulate deltas
            for eta in etas:
                loss_before, loss_after, delta_loss = results[eta]
                avg_deltas[eta] += delta_loss / num_seeds

        # Check that delta/eta is constant (linear scaling)
        ratios = [avg_deltas[eta] / eta for eta in etas]
        assert math.isclose(
            max(ratios), min(ratios), rel_tol=0.05
        ), f"Delta/eta ratios not constant: {dict(zip(etas, ratios))}"
