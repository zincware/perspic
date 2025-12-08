import pytest
import torch
import torch.nn as nn

from perspic.calculator.linearizer import Linearizer


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
    """Tests for Linearizer functionality."""

    def test_simple_model_matches_analytical_gradient(self, simple_model):
        """Test that linearizer results match analytical gradient norm squared."""
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

        # Use Linearizer
        simple_model.load_state_dict(initial_state)
        simple_model.zero_grad()
        results = Linearizer().compute(
            model=simple_model,
            criterion=criterion,
            x1=x,
            y1=y,
        )

        # Verify gradient norm squared matches
        _, _, delta_loss = results["self"]
        computed_grad_norm = -delta_loss

        assert (
            abs(computed_grad_norm - grad_norm_squared) < 1e-6
        ), f"Expected {grad_norm_squared}, got {computed_grad_norm}"

    def test_cross_response_integration(self, simple_model):
        """Test cross-response with different batches."""
        torch.manual_seed(42)
        criterion = nn.MSELoss()
        x1 = torch.randn(8, 10)
        y1 = torch.randn(8, 5)
        x2 = torch.randn(8, 10)
        y2 = torch.randn(8, 5)

        results = Linearizer().compute(
            model=simple_model,
            criterion=criterion,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        # Both self and cross should be present
        assert "self" in results
        assert "cross" in results

        # Self-response should be negative (gradient norm squared)
        _, _, delta_self = results["self"]
        assert delta_self < 0

        # Cross-response can be positive or negative depending on gradient alignment
        _, _, delta_cross = results["cross"]
        assert isinstance(delta_cross, float)
