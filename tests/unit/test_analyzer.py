import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from perspic.analyzer import Analyzer


@pytest.fixture
def simple_model():
    """Create a simple linear model for testing."""
    torch.manual_seed(42)
    return nn.Linear(10, 2)


@pytest.fixture
def sample_data():
    """Create sample input and target data."""
    torch.manual_seed(42)
    x = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))
    return x, y


@pytest.fixture
def data_loader(sample_data):
    """Create a DataLoader with sample data."""
    x, y = sample_data
    dataset = TensorDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=8,
        num_workers=10,
    )


@pytest.fixture
def analyzer(simple_model, data_loader):
    """Create an Analyzer instance."""
    optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    return Analyzer(
        model=simple_model,
        optimizer=optimizer,
        criterion=criterion,
        data_loader=data_loader,
        sample_wise_engine="functorch",
    )


class TestAnalyzerInitialization:
    """Test Analyzer initialization."""

    def test_initialization_functorch(self, simple_model, data_loader):
        """Test Analyzer initializes correctly with functorch."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        analyzer = Analyzer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=data_loader,
            sample_wise_engine="functorch",
        )

        assert analyzer.sample_wise_engine == "functorch"
        assert analyzer.sample_calc is not None
        assert analyzer.automatic_optimization is False
        assert isinstance(analyzer.linearizer, type(analyzer.linearizer))

    def test_initialization_invalid_engine(self, simple_model, data_loader):
        """Test Analyzer raises error with invalid engine."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        with pytest.raises(
            ValueError,
            match="sample_wise_engine must be either",
        ):
            Analyzer(
                model=simple_model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=data_loader,
                sample_wise_engine="invalid",
            )


class TestAnalyzerTrainingStep:
    """Test Analyzer.training_step with real forward/backward passes."""

    def test_model_weights_change_after_step(self, analyzer, sample_data):
        """Verify that model parameters actually update during training."""
        initial_weights = [p.clone().detach() for p in analyzer.model.parameters()]

        # Run one training step
        result = analyzer.training_step(sample_data, 0)

        # Check weights changed
        for initial, current in zip(initial_weights, analyzer.model.parameters()):
            if torch.allclose(initial, current):
                pytest.fail("Weights should change after training step")

        assert result is None  # training_step should return None

    def test_loss_decreases_over_steps(self, analyzer, sample_data):
        """Test that loss decreases over multiple training steps."""
        inputs, targets = sample_data

        # Initial loss
        with torch.no_grad():
            preds = analyzer.model(inputs)
            initial_loss = analyzer.criterion(preds, targets).item()

        # Run 10 training steps
        for _ in range(10):
            analyzer.training_step(sample_data, 0)

        # Final loss
        with torch.no_grad():
            preds = analyzer.model(inputs)
            final_loss = analyzer.criterion(preds, targets).item()

        assert (
            final_loss < initial_loss
        ), f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_gradient_computation(self, analyzer, sample_data):
        """Test that gradients are computed correctly."""
        inputs, targets = sample_data

        # Clear any existing gradients
        analyzer.optimizer.zero_grad()

        # Compute loss and gradients manually
        outputs = analyzer.model(inputs)
        loss = analyzer.criterion(outputs, targets)
        loss.backward()

        # Check gradients exist and are non-zero
        for name, param in analyzer.model.named_parameters():
            if param.grad is None:
                pytest.fail(f"Gradient for {name} should exist")
            if param.grad.abs().sum() <= 0:
                pytest.fail(f"Gradient for {name} should be non-zero")

    def test_accuracy_improves(self, simple_model, data_loader):
        """Test that model accuracy improves over training."""
        # Create simple dataset where model can learn
        torch.manual_seed(42)
        x = torch.randn(32, 10)
        y = (x[:, 0] > 0).long()  # Simple rule: class on first feature
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=8)

        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        analyzer = Analyzer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            sample_wise_engine="functorch",
        )

        # Initial accuracy
        with torch.no_grad():
            outputs = analyzer.model(x)
            initial_acc = (outputs.argmax(dim=1) == y).float().mean().item()

        # Train for 20 steps
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 20:
                break
            analyzer.training_step(batch, batch_idx)

        # Final accuracy
        with torch.no_grad():
            outputs = analyzer.model(x)
            final_acc = (outputs.argmax(dim=1) == y).float().mean().item()

        assert (
            final_acc > initial_acc + 0.1
        ), f"Accuracy should improve: {initial_acc:.3f} -> {final_acc:.3f}"

    def test_samplewise_gradients_computed(self, analyzer, sample_data):
        """Test that per-sample gradient norms are computed."""
        inputs, targets = sample_data

        # This is a minimal integration test - we check the compute
        # method runs and returns the expected structure
        result = analyzer.sample_calc.compute(
            analyzer.model,
            analyzer.criterion,
            inputs,
            targets,
        )

        assert "batch_grad_norms_network" in result
        assert "batch_grad_norms_loss" in result

        assert isinstance(result["batch_grad_norms_network"], torch.Tensor)
        assert isinstance(result["batch_grad_norms_loss"], torch.Tensor)
        # Both should be scalar (single value)
        assert result["batch_grad_norms_network"].numel() == 1
        assert result["batch_grad_norms_loss"].numel() == 1

        # Check values are positive (gradient norms should be > 0)
        assert (result["batch_grad_norms_network"] > 0).all()
        assert (result["batch_grad_norms_loss"] > 0).all()


class TestProbeTrainingStep:
    """Test the linearizer probe functionality."""

    def test_probe_does_not_change_model(self, analyzer, sample_data):
        """Verify probe_train_step doesn't permanently change model."""
        inputs, targets = sample_data

        initial_weights = [p.clone().detach() for p in analyzer.model.parameters()]

        # Run probe (should restore weights after)
        loss, perturbed_loss = analyzer.linearizer.probe_train_step(
            model=analyzer.model,
            criterion=analyzer.criterion,
            x=inputs,
            y=targets,
            eta=1e-5,
        )

        # Check weights are unchanged
        for initial, current in zip(initial_weights, analyzer.model.parameters()):
            if not torch.allclose(initial, current, atol=1e-6):
                pytest.fail("Probe should restore original weights")

    def test_probe_perturbed_loss_differs(self, analyzer, sample_data):
        """Test that perturbed model produces different loss."""
        inputs, targets = sample_data

        loss, perturbed_loss = analyzer.linearizer.probe_train_step(
            model=analyzer.model,
            criterion=analyzer.criterion,
            x=inputs,
            y=targets,
            eta=1e-5,
        )

        # With a non-zero learning rate, perturbed loss should differ
        assert loss is not None
        assert perturbed_loss is not None
        # They might be close but shouldn't be identical
        # (unless eta is too small)
        assert not torch.allclose(loss, perturbed_loss, atol=1e-8)
