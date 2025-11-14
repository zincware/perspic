import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from perspic.analyzer import Analyzer, PerspicTrainer


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


class TestEndToEndTraining:
    """Integration tests for full training pipeline."""

    def test_full_training_epoch(self, simple_model, data_loader):
        """Test complete training epoch runs successfully."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        analyzer = Analyzer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=data_loader,
            sample_wise_engine="functorch",
        )

        # Run one epoch
        for batch_idx, batch in enumerate(data_loader):
            result = analyzer.training_step(batch, batch_idx)
            assert result is None


class TestTrainerIntegration:
    """Test integration of PerspicTrainer with Analyzer."""

    def test_trainer_runs_epoch(self, simple_model, data_loader):
        """Test that PerspicTrainer runs epoch with Analyzer."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        analyzer = Analyzer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=data_loader,
            sample_wise_engine="functorch",
        )

        trainer = PerspicTrainer(
            analyzer=analyzer,
            max_epochs=1,
            log_every_n_steps=1,
        )

        trainer.fit()

        # After fitting, ensure model parameters have changed
        for param in simple_model.parameters():
            if param.grad is None:
                pytest.fail("Parameter gradients should exist after training")
            if param.grad.abs().sum() <= 0:
                pytest.fail("Params grads should be non-zero after training")
