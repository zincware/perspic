import pytest
import torch
import torch.nn as nn

from perspic.calculator.linearizer import Linearizer
from perspic.utils import BatchStatSnapshot


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


@pytest.fixture
def complex_model():
    """Create a more complex model with BatchNorm."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrize tests to run on both CPU and GPU."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


class TestLinearizer:
    """Tests for the Linearizer class."""

    def test_linearizer_returns_correct_format(self, simple_model):
        """Test that Linearizer returns the correct format with 'self' and 'cross' keys."""
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        linearizer = Linearizer()
        results = linearizer.compute(
            model=simple_model,
            criterion=criterion,
            x1=x,
            y1=y,
        )

        assert "self" in results
        assert "cross" in results
        assert results["cross"] is None  # No cross batch provided
        loss, perturbed_loss, delta_loss = results["self"]
        assert isinstance(loss, float)
        assert isinstance(perturbed_loss, float)
        assert isinstance(delta_loss, float)
        # delta_loss should be -||grad||^2 (negative)
        assert delta_loss <= 0

    def test_linearizer_matches_manual_grad_norm_simple_model(self, simple_model):
        """Test that Linearizer computes the correct gradient norm squared."""
        torch.manual_seed(42)
        criterion = nn.MSELoss()
        x = torch.randn(8, 10)
        y = torch.randn(8, 5)

        # Manually compute gradient norm squared
        simple_model.zero_grad()
        loss = criterion(simple_model(x), y)
        loss.backward()
        expected_grad_norm_squared = sum(
            (p.grad**2).sum().item()
            for p in simple_model.parameters()
            if p.grad is not None
        )

        # Reset and use Linearizer
        simple_model.zero_grad()
        linearizer = Linearizer()
        results = linearizer.compute(
            model=simple_model,
            criterion=criterion,
            x1=x,
            y1=y,
        )

        loss_val, perturbed_loss, delta_loss = results["self"]
        computed_grad_norm_squared = -delta_loss

        assert (
            abs(computed_grad_norm_squared - expected_grad_norm_squared) < 1e-6
        ), f"Expected {expected_grad_norm_squared}, got {computed_grad_norm_squared}"

    def test_linearizer_matches_manual_grad_norm_complex_model(self, complex_model):
        """Test that Linearizer computes the correct gradient norm squared
        for a more complex model with BatchNorm."""
        torch.manual_seed(42)
        criterion = nn.MSELoss()
        x = torch.randn(8, 10)
        y = torch.randn(8, 5)

        # Manually compute gradient norm squared
        complex_model.zero_grad()
        loss = criterion(complex_model(x), y)
        loss.backward()
        expected_grad_norm_squared = sum(
            (p.grad**2).sum().item()
            for p in complex_model.parameters()
            if p.grad is not None
        )
        # Reset and use Linearizer
        complex_model.zero_grad()
        linearizer = Linearizer()
        results = linearizer.compute(
            model=complex_model,
            criterion=criterion,
            x1=x,
            y1=y,
        )
        loss_val, perturbed_loss, delta_loss = results["self"]
        computed_grad_norm_squared = -delta_loss
        assert (
            abs(computed_grad_norm_squared - expected_grad_norm_squared) < 1e-6
        ), f"Expected {expected_grad_norm_squared}, got {computed_grad_norm_squared}"

    def test_linearizer_matches_manual_grad_norm_with_snapshot(self, complex_model):
        """Test that Linearizer computes the correct gradient norm squared
        when using BatchStatSnapshot with BatchNorm layers."""
        torch.manual_seed(42)
        criterion = nn.MSELoss()
        x = torch.randn(8, 10)
        y = torch.randn(8, 5)

        with BatchStatSnapshot(complex_model, x):
            # Manually compute gradient norm squared
            complex_model.zero_grad()
            loss = criterion(complex_model(x), y)
            loss.backward()
            expected_grad_norm_squared = sum(
                (p.grad**2).sum().item()
                for p in complex_model.parameters()
                if p.grad is not None
            )

            # Reset and use Linearizer
            complex_model.zero_grad()
            linearizer = Linearizer()
            results = linearizer.compute(
                model=complex_model,
                criterion=criterion,
                x1=x,
                y1=y,
            )
            loss_val, perturbed_loss, delta_loss = results["self"]
            computed_grad_norm_squared = -delta_loss
            assert (
                abs(computed_grad_norm_squared - expected_grad_norm_squared) < 1e-6
            ), f"Expected {expected_grad_norm_squared}, got {computed_grad_norm_squared}"

    def test_linearizer_perturbed_loss_formula(self, simple_model):
        """Test that perturbed_loss = loss - ||grad||^2."""
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        linearizer = Linearizer()
        results = linearizer.compute(
            model=simple_model,
            criterion=criterion,
            x1=x,
            y1=y,
        )

        loss, perturbed_loss, delta_loss = results["self"]
        assert abs(perturbed_loss - (loss + delta_loss)) < 1e-7

    def test_linearizer_gradients_zeroed(self, simple_model):
        """Test that gradients are zeroed after compute."""
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        linearizer = Linearizer()
        linearizer.compute(model=simple_model, criterion=criterion, x1=x, y1=y)

        for param in simple_model.parameters():
            if param.grad is not None:
                assert torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                ), "Gradients should be zeroed after compute"


class TestLinearizerCrossResponse:
    """Tests for the cross-response functionality of Linearizer."""

    def test_cross_response_returns_both_keys(self, simple_model):
        """Test that compute with x2, y2 returns both 'self' and 'cross' keys."""
        criterion = nn.CrossEntropyLoss()
        x1 = torch.randn(4, 10)
        y1 = torch.randint(0, 5, (4,))
        x2 = torch.randn(4, 10)
        y2 = torch.randint(0, 5, (4,))

        linearizer = Linearizer()
        results = linearizer.compute(
            model=simple_model,
            criterion=criterion,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        assert "self" in results
        assert "cross" in results
        assert len(results) == 2

        # Check self result
        loss_1, perturbed_1, delta_1 = results["self"]
        assert isinstance(loss_1, float)
        assert isinstance(perturbed_1, float)
        assert isinstance(delta_1, float)
        assert delta_1 <= 0  # -||grad||^2 is always negative

        # Check cross result
        loss_2, perturbed_2, delta_2 = results["cross"]
        assert isinstance(loss_2, float)
        assert isinstance(perturbed_2, float)
        assert isinstance(delta_2, float)

    def test_cross_response_matches_manual_dot_product(self, simple_model):
        """Test that cross-response matches manually computed gradient dot product."""
        torch.manual_seed(42)
        criterion = nn.MSELoss()
        x1 = torch.randn(4, 10)
        y1 = torch.randn(4, 5)
        x2 = torch.randn(4, 10)
        y2 = torch.randn(4, 5)

        # Manually compute gradient dot product
        simple_model.zero_grad()
        loss_1 = criterion(simple_model(x1), y1)
        loss_1.backward()
        grads_1 = [
            p.grad.clone() for p in simple_model.parameters() if p.grad is not None
        ]

        simple_model.zero_grad()
        loss_2 = criterion(simple_model(x2), y2)
        loss_2.backward()
        grads_2 = [
            p.grad.clone() for p in simple_model.parameters() if p.grad is not None
        ]

        expected_dot_product = sum(
            (g1 * g2).sum().item() for g1, g2 in zip(grads_1, grads_2)
        )

        # Use Linearizer
        simple_model.zero_grad()
        linearizer = Linearizer()
        results = linearizer.compute(
            model=simple_model,
            criterion=criterion,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        _, _, delta_cross = results["cross"]
        computed_dot_product = -delta_cross

        assert (
            abs(computed_dot_product - expected_dot_product) < 1e-5
        ), f"Expected {expected_dot_product}, got {computed_dot_product}"

    def test_cross_response_same_batch_equals_self_response(self, simple_model):
        """Test that cross-response with same batch equals self-response."""
        torch.manual_seed(42)
        criterion = nn.MSELoss()
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        linearizer = Linearizer()
        results = linearizer.compute(
            model=simple_model,
            criterion=criterion,
            x1=x,
            y1=y,
            x2=x,
            y2=y,
        )

        _, _, delta_self = results["self"]
        _, _, delta_cross = results["cross"]

        # When x1==x2 and y1==y2, cross-response should equal self-response
        assert (
            abs(delta_self - delta_cross) < 1e-6
        ), f"With same batch, self ({delta_self}) should equal cross ({delta_cross})"

    def test_cross_response_loss_values(self, simple_model):
        """Test that loss values in cross response are computed on second batch."""
        torch.manual_seed(42)
        criterion = nn.MSELoss()
        x1 = torch.randn(4, 10)
        y1 = torch.randn(4, 5)
        x2 = torch.randn(4, 10)
        y2 = torch.randn(4, 5)

        # Compute expected loss on x2, y2
        with torch.no_grad():
            expected_loss_2 = criterion(simple_model(x2), y2).item()

        linearizer = Linearizer()
        results = linearizer.compute(
            model=simple_model,
            criterion=criterion,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        loss_2, _, _ = results["cross"]
        assert (
            abs(loss_2 - expected_loss_2) < 1e-6
        ), f"Loss in cross should be on second batch: expected {expected_loss_2}, got {loss_2}"

    def test_cross_response_gradients_zeroed(self, simple_model):
        """Test that gradients are zeroed after compute with cross-response."""
        criterion = nn.CrossEntropyLoss()
        x1 = torch.randn(4, 10)
        y1 = torch.randint(0, 5, (4,))
        x2 = torch.randn(4, 10)
        y2 = torch.randint(0, 5, (4,))

        linearizer = Linearizer()
        linearizer.compute(
            model=simple_model,
            criterion=criterion,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        for param in simple_model.parameters():
            if param.grad is not None:
                assert torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                ), "Gradients should be zeroed after compute"
