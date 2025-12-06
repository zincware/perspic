"""Integration tests for the analyzer module."""

import copy
import os

import psutil
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from perspic.analyzer import analyzer


# Test Fixtures
@pytest.fixture
def simple_lightning_module():
    """Create a simple LightningModule for testing."""

    class SimpleLightningModule(pl.LightningModule):
        def __init__(self, lr=0.001):
            super().__init__()
            self.save_hyperparameters()
            self.model = nn.Linear(10, 2)
            self.criterion = F.cross_entropy

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    return SimpleLightningModule


@pytest.fixture
def batchnorm_module():
    """Create a module with BatchNorm layers."""

    class BatchNormModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(10, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Linear(10, 2),
            )
            self.criterion = F.cross_entropy

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    return BatchNormModule


@pytest.fixture
def dropout_residual_module():
    """Create a module with Dropout and residual connections."""

    class ResidualBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dim, dim)
            )

        def forward(self, x):
            return x + self.block(x)

    class DropoutResidualModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(10, 20),
                ResidualBlock(20),
                ResidualBlock(20),
                nn.Linear(20, 2),
            )
            self.criterion = F.cross_entropy

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    return DropoutResidualModule


@pytest.fixture
def simple_dataloader():
    """Create a simple DataLoader."""
    torch.manual_seed(42)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def variable_batch_dataloader():
    """Create a DataLoader with variable batch sizes (last batch smaller)."""
    torch.manual_seed(42)
    x = torch.randn(35, 10)  # 35 samples -> batches of 8, 8, 8, 8, 3
    y = torch.randint(0, 2, (35,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8, shuffle=False)


# Test Classes
class TestTrainingTrajectoryPreservation:
    """Critical tests: Verify analyzer doesn't change training trajectory."""

    @pytest.mark.parametrize("sample_wise_engine", ["functorch", "opacus"])
    def test_enabled_analyzer_matches_disabled_single_step(
        self, simple_lightning_module, sample_wise_engine
    ):
        """Enabled analyzer produces same loss as disabled (single step)."""
        torch.manual_seed(42)
        model_disabled = analyzer(
            simple_lightning_module,
            sample_wise_engine=sample_wise_engine,
            disable_analyzer=True,
        )

        torch.manual_seed(42)
        model_enabled = analyzer(
            simple_lightning_module,
            sample_wise_engine=sample_wise_engine,
            disable_analyzer=False,
        )

        model_initial_ref = copy.deepcopy(model_enabled)

        # Same batch
        torch.manual_seed(100)
        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        # Use Trainer for both
        trainer_disabled = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer_enabled = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer_disabled.fit(model_disabled, dataloader)
        trainer_enabled.fit(model_enabled, dataloader)

        # Assert that model states have changed
        for p_ref, p_dis in zip(
            model_initial_ref.parameters(), model_disabled.parameters()
        ):
            assert not torch.equal(
                p_ref, p_dis
            ), "Disabled analyzer model did not train!"

        # Compare model states
        for p_dis, p_en in zip(model_disabled.parameters(), model_enabled.parameters()):
            assert torch.allclose(p_dis, p_en, atol=1e-5), "Parameters diverged!"

    @pytest.mark.parametrize("sample_wise_engine", ["functorch", "opacus"])
    def test_enabled_analyzer_matches_disabled_multi_step(
        self, simple_lightning_module, sample_wise_engine
    ):
        """Enabled analyzer produces same loss as disabled (multiple steps)."""
        torch.manual_seed(42)
        model_disabled = analyzer(
            simple_lightning_module,
            sample_wise_engine=sample_wise_engine,
            disable_analyzer=True,
        )

        torch.manual_seed(42)
        model_enabled = analyzer(
            simple_lightning_module,
            sample_wise_engine=sample_wise_engine,
            disable_analyzer=False,
        )

        model_initial_ref = copy.deepcopy(model_enabled)

        # Same batch
        torch.manual_seed(100)
        batch = (torch.randn(8, 10), torch.randint(0, 2, (8,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=8)

        # Use Trainer for both
        trainer_disabled = pl.Trainer(
            max_steps=5,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer_enabled = pl.Trainer(
            max_steps=5,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer_disabled.fit(model_disabled, dataloader)
        trainer_enabled.fit(model_enabled, dataloader)

        # Assert that model states have changed
        for p_ref, p_dis in zip(
            model_initial_ref.parameters(), model_disabled.parameters()
        ):
            assert not torch.equal(
                p_ref, p_dis
            ), "Disabled analyzer model did not train!"

        # Compare model states
        for p_dis, p_en in zip(model_disabled.parameters(), model_enabled.parameters()):
            assert torch.allclose(p_dis, p_en, atol=1e-5), "Parameters diverged!"


class TestAnalyzerEndToEnd:
    """Complete training workflow tests."""

    def test_single_training_step_updates_parameters(self, simple_lightning_module):
        """A single training step actually updates model parameters."""
        torch.manual_seed(42)
        model = analyzer(simple_lightning_module)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Run one training step
        torch.manual_seed(100)
        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader)

        # Parameters should have changed
        for p_initial, p_current in zip(initial_params, model.parameters()):
            assert not torch.equal(
                p_initial, p_current
            ), "Parameters didn't update after training step"

    def test_multi_step_training_converges(self, simple_lightning_module):
        """Loss decreases over multiple training steps on simple problem."""
        torch.manual_seed(42)
        model = analyzer(simple_lightning_module)

        # Fixed simple problem: same batch repeatedly
        torch.manual_seed(100)
        batch = (torch.randn(8, 10), torch.randint(0, 2, (8,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=8)

        # Custom callback to track losses
        class LossTracker(pl.Callback):
            def __init__(self):
                self.losses = []

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.losses.append(outputs["loss"].item())

        loss_tracker = LossTracker()
        trainer = pl.Trainer(
            max_epochs=20,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[loss_tracker],
        )
        trainer.fit(model, dataloader)

        # Loss should decrease (at least last loss < first loss)
        assert (
            loss_tracker.losses[-1] < loss_tracker.losses[0]
        ), f"Loss didn't decrease: {loss_tracker.losses[0]} -> {loss_tracker.losses[-1]}"


class TestModelCompatibility:
    """Test with different model architectures."""

    @pytest.mark.parametrize("sample_wise_engine", ["functorch", "opacus"])
    def test_with_simple_mlp(self, simple_lightning_module, sample_wise_engine):
        """Works with simple MLP."""
        torch.manual_seed(42)
        model = analyzer(simple_lightning_module, sample_wise_engine=sample_wise_engine)

        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader)

        # Should complete without errors
        assert True

    @pytest.mark.parametrize("sample_wise_engine", ["functorch", "opacus"])
    def test_with_batchnorm_model(self, batchnorm_module, sample_wise_engine):
        """Works with BatchNorm layers."""
        torch.manual_seed(42)
        model = analyzer(batchnorm_module, sample_wise_engine=sample_wise_engine)

        # Train mode
        model.train()
        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = pl.Trainer(
            max_steps=2,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader)

        # Verify model is still in train mode after analysis
        assert model.training

        # Should complete without errors
        assert True

    @pytest.mark.parametrize("sample_wise_engine", ["functorch", "opacus"])
    def test_with_dropout_residual_model(
        self, dropout_residual_module, sample_wise_engine
    ):
        """Works with Dropout and residual connections."""
        torch.manual_seed(42)
        model = analyzer(dropout_residual_module, sample_wise_engine=sample_wise_engine)

        model.train()
        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = pl.Trainer(
            max_steps=2,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader)

        # Verify model is still in train mode after analysis
        assert model.training

        # Should complete without errors
        assert True


class TestPyTorchIntegration:
    """Integration with PyTorch components."""

    @pytest.mark.parametrize("optimizer_class", [torch.optim.Adam, torch.optim.SGD])
    def test_with_different_optimizers(self, simple_lightning_module, optimizer_class):
        """Works with different optimizers."""

        # Modify the module to use the specified optimizer
        class CustomOptModule(pl.LightningModule):
            def __init__(self, lr=0.001):
                super().__init__()
                self.save_hyperparameters()
                self.model = nn.Linear(10, 2)
                self.criterion = F.cross_entropy
                self.optimizer_class = optimizer_class

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                return loss

            def configure_optimizers(self):
                return self.optimizer_class(self.parameters(), lr=self.hparams.lr)

        torch.manual_seed(42)
        model = analyzer(CustomOptModule)

        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader)

        # Should complete without errors
        assert True

    def test_with_scheduler(self, simple_lightning_module):
        """Works with learning rate scheduler."""

        # Modify module to include scheduler
        class SchedulerModule(pl.LightningModule):
            def __init__(self, lr=0.1):
                super().__init__()
                self.save_hyperparameters()
                self.model = nn.Linear(10, 2)
                self.criterion = F.cross_entropy

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                return loss

            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.1
                )
                return {"optimizer": optimizer, "lr_scheduler": scheduler}

        torch.manual_seed(42)
        model = analyzer(SchedulerModule)

        # Train for several steps
        batch = (torch.randn(40, 10), torch.randint(0, 2, (40,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = pl.Trainer(
            max_steps=10,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader)

        # Should complete without errors
        assert True

    def test_with_variable_batch_sizes(
        self, simple_lightning_module, variable_batch_dataloader
    ):
        """Handles variable batch sizes correctly."""

        # Track batch sizes with callback
        class BatchSizeTracker(pl.Callback):
            def __init__(self):
                self.batch_sizes = []

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                self.batch_sizes.append(batch[0].shape[0])

        torch.manual_seed(42)
        model = analyzer(simple_lightning_module)

        tracker = BatchSizeTracker()
        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[tracker],
        )
        trainer.fit(model, variable_batch_dataloader)

        # Should have batches of size [8, 8, 8, 8, 3]
        assert tracker.batch_sizes == [8, 8, 8, 8, 3]


class TestMemoryAndDevices:
    """Memory management and device handling."""

    @pytest.mark.parametrize("sample_wise_engine", ["functorch", "opacus"])
    def test_no_memory_leak(self, simple_lightning_module, sample_wise_engine):
        """No memory leak over multiple training steps."""

        torch.manual_seed(42)
        model = analyzer(simple_lightning_module, sample_wise_engine=sample_wise_engine)

        # Run many steps
        batch = (torch.randn(200, 10), torch.randint(0, 2, (200,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        # Track memory usage with callback
        class MemoryTracker(pl.Callback):
            def __init__(self):
                self.memory_usage = []
                self.process = psutil.Process(os.getpid())

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                # Force garbage collection and measure
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                mem_mb = self.process.memory_info().rss / 1024 / 1024
                self.memory_usage.append(mem_mb)

        memory_tracker = MemoryTracker()
        trainer = pl.Trainer(
            max_steps=50,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[memory_tracker],
        )
        trainer.fit(model, dataloader)

        # Check memory didn't grow significantly
        # Allow some growth for initial allocations, but should stabilize
        initial_mem = memory_tracker.memory_usage[5]  # After warmup
        final_mem = memory_tracker.memory_usage[-1]
        mem_growth_mb = final_mem - initial_mem

        # Memory should not grow more than 1 MB over 45 steps
        assert mem_growth_mb < 1, (
            f"Memory leak detected: grew {mem_growth_mb:.2f} MB "
            f"from {initial_mem:.2f} MB to {final_mem:.2f} MB"
        )

    @pytest.mark.parametrize("sample_wise_engine", ["functorch", "opacus"])
    def test_cpu_training(self, simple_lightning_module, sample_wise_engine):
        """Works on CPU."""
        torch.manual_seed(42)
        model = analyzer(simple_lightning_module, sample_wise_engine=sample_wise_engine)
        model = model.to("cpu")

        batch = (torch.randn(4, 10).to("cpu"), torch.randint(0, 2, (4,)).to("cpu"))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader)

        # Check that parameters are on CPU
        for param in model.parameters():
            assert param.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("sample_wise_engine", ["functorch", "opacus"])
    def test_gpu_training(self, simple_lightning_module, sample_wise_engine):
        """Works on GPU."""
        torch.manual_seed(42)
        model = analyzer(simple_lightning_module, sample_wise_engine=sample_wise_engine)

        # Don't manually move to GPU - let Trainer handle it
        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        # Track device during training with callback
        class DeviceTracker(pl.Callback):
            def __init__(self):
                self.devices_seen = []

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                # Check model device during training
                for param in pl_module.parameters():
                    self.devices_seen.append(param.device.type)
                    break  # Just check first param

        device_tracker = DeviceTracker()

        # Use accelerator="gpu" to tell Trainer to use GPU
        trainer = pl.Trainer(
            max_steps=1,
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[device_tracker],
        )
        trainer.fit(model, dataloader)

        # Check that training happened on CUDA
        assert all(
            dev == "cuda" for dev in device_tracker.devices_seen
        ), f"Expected training on CUDA but saw devices: {device_tracker.devices_seen}"


class TestManualOptimization:
    """Test delegation to manual optimization modules."""

    def test_delegates_to_manual_optimization_module(self):
        """Correctly delegates optimization when wrapped module uses manual optimization."""

        class ManualOptModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(10, 2)
                self.criterion = F.cross_entropy
                self.automatic_optimization = False
                self.step_called = False

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                opt = self.optimizers()
                opt.zero_grad()
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                self.manual_backward(loss)
                opt.step()
                self.step_called = True  # Track that parent's step was called
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)

        with pytest.warns(UserWarning, match="manual optimization"):
            torch.manual_seed(42)
            model = analyzer(ManualOptModule)

        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, dataloader)

        # Verify delegation happened
        assert model.delegate_optimization is True


class TestSamplewiseEngineConsistency:
    """Tests that functorch and opacus engines produce consistent results."""

    def test_engines_produce_same_gradient_norms_simple_mlp(self):
        """Both engines compute the same gradient norms for a simple MLP."""

        class SimpleMLP(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 2),
                )
                self.criterion = F.cross_entropy

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)

        # Track logged metrics with callback
        class MetricsTracker(pl.Callback):
            def __init__(self):
                self.metrics = []

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                # Capture the logged metrics
                self.metrics.append(
                    {
                        "chi_net": trainer.callback_metrics.get("chi_net"),
                        "chi_loss": trainer.callback_metrics.get("chi_loss"),
                    }
                )

        # Run with functorch
        torch.manual_seed(42)
        model_functorch = analyzer(SimpleMLP, sample_wise_engine="functorch")
        tracker_functorch = MetricsTracker()

        torch.manual_seed(100)
        batch = (torch.randn(8, 10), torch.randint(0, 2, (8,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=8)

        trainer_functorch = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[tracker_functorch],
        )
        trainer_functorch.fit(model_functorch, dataloader)

        # Run with opacus (same seed for identical model initialization)
        torch.manual_seed(42)
        model_opacus = analyzer(SimpleMLP, sample_wise_engine="opacus")
        tracker_opacus = MetricsTracker()

        # Use same dataloader (recreate with same seed)
        torch.manual_seed(100)
        batch = (torch.randn(8, 10), torch.randint(0, 2, (8,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=8)

        trainer_opacus = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[tracker_opacus],
        )
        trainer_opacus.fit(model_opacus, dataloader)

        # Compare gradient norms
        functorch_network = tracker_functorch.metrics[0]["chi_net"]
        opacus_network = tracker_opacus.metrics[0]["chi_net"]
        functorch_loss = tracker_functorch.metrics[0]["chi_loss"]
        opacus_loss = tracker_opacus.metrics[0]["chi_loss"]

        assert torch.allclose(functorch_network, opacus_network, atol=1e-5), (
            f"Network gradient norms differ: functorch={functorch_network}, "
            f"opacus={opacus_network}"
        )
        assert torch.allclose(functorch_loss, opacus_loss, atol=1e-5), (
            f"Loss gradient norms differ: functorch={functorch_loss}, "
            f"opacus={opacus_loss}"
        )

    def test_engines_produce_same_gradient_norms_batchnorm(self):
        """Both engines compute the same gradient norms for a model with BatchNorm."""

        class BatchNormMLP(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.BatchNorm1d(20),
                    nn.ReLU(),
                    nn.Linear(20, 2),
                )
                self.criterion = F.cross_entropy

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)

        # Track logged metrics with callback
        class MetricsTracker(pl.Callback):
            def __init__(self):
                self.metrics = []

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.metrics.append(
                    {
                        "chi_net": trainer.callback_metrics.get("chi_net"),
                        "chi_loss": trainer.callback_metrics.get("chi_loss"),
                    }
                )

        # Run with functorch
        torch.manual_seed(42)
        model_functorch = analyzer(BatchNormMLP, sample_wise_engine="functorch")
        tracker_functorch = MetricsTracker()

        torch.manual_seed(100)
        batch = (torch.randn(8, 10), torch.randint(0, 2, (8,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=8)

        trainer_functorch = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[tracker_functorch],
        )
        trainer_functorch.fit(model_functorch, dataloader)

        # Run with opacus
        torch.manual_seed(42)
        model_opacus = analyzer(BatchNormMLP, sample_wise_engine="opacus")
        tracker_opacus = MetricsTracker()

        torch.manual_seed(100)
        batch = (torch.randn(8, 10), torch.randint(0, 2, (8,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=8)

        trainer_opacus = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[tracker_opacus],
        )
        trainer_opacus.fit(model_opacus, dataloader)

        # Compare gradient norms
        functorch_network = tracker_functorch.metrics[0]["chi_net"]
        opacus_network = tracker_opacus.metrics[0]["chi_net"]
        functorch_loss = tracker_functorch.metrics[0]["chi_loss"]
        opacus_loss = tracker_opacus.metrics[0]["chi_loss"]

        assert torch.allclose(functorch_network, opacus_network, atol=1e-5), (
            f"Network gradient norms differ: functorch={functorch_network}, "
            f"opacus={opacus_network}"
        )
        assert torch.allclose(functorch_loss, opacus_loss, atol=1e-5), (
            f"Loss gradient norms differ: functorch={functorch_loss}, "
            f"opacus={opacus_loss}"
        )

    def test_engines_produce_same_training_trajectory(self):
        """Both engines result in identical model parameters after training."""

        class SimpleMLP(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(10, 2)
                self.criterion = F.cross_entropy

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                return loss

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        # Train with functorch
        torch.manual_seed(42)
        model_functorch = analyzer(SimpleMLP, sample_wise_engine="functorch")

        torch.manual_seed(100)
        batch = (torch.randn(8, 10), torch.randint(0, 2, (8,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=8)

        trainer_functorch = pl.Trainer(
            max_steps=5,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer_functorch.fit(model_functorch, dataloader)

        # Train with opacus
        torch.manual_seed(42)
        model_opacus = analyzer(SimpleMLP, sample_wise_engine="opacus")

        torch.manual_seed(100)
        batch = (torch.randn(8, 10), torch.randint(0, 2, (8,)))
        dataset = TensorDataset(*batch)
        dataloader = DataLoader(dataset, batch_size=8)

        trainer_opacus = pl.Trainer(
            max_steps=5,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer_opacus.fit(model_opacus, dataloader)

        # Compare final model parameters
        for p_functorch, p_opacus in zip(
            model_functorch.parameters(), model_opacus.parameters()
        ):
            assert torch.allclose(
                p_functorch, p_opacus, atol=1e-5
            ), "Model parameters diverged between functorch and opacus engines"


class TestAnalyzerWithScheduling:
    """Integration tests for analyzer with scheduling options."""

    @pytest.fixture
    def simple_module(self):
        """Create a simple LightningModule for scheduling tests."""

        class SimpleModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(10, 2)
                self.criterion = F.cross_entropy

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                return self.criterion(self(x), y)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        return SimpleModule

    @pytest.fixture
    def simple_dataloader(self):
        """Create a simple dataloader."""
        torch.manual_seed(42)
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        return DataLoader(TensorDataset(x, y), batch_size=4)

    def test_training_with_analyze_every(self, simple_module, simple_dataloader):
        """Test training completes with analyze_every."""
        model = analyzer(simple_module, analyze_every=2)

        trainer = pl.Trainer(
            max_steps=8,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, simple_dataloader)

        assert trainer.global_step == 8

    def test_training_with_logarithmic_schedule(self, simple_module, simple_dataloader):
        """Test training completes with logarithmic schedule."""
        from perspic.logger import logarithmic_windows

        schedule = logarithmic_windows(max_steps=10, base_window=2)
        model = analyzer(simple_module, analysis_schedule=schedule)

        trainer = pl.Trainer(
            max_steps=8,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, simple_dataloader)

        assert trainer.global_step == 8

    def test_analyze_every_reduces_compute(self, simple_module, simple_dataloader):
        """Test analyze_every reduces analysis calls."""
        from unittest.mock import MagicMock, patch

        model = analyzer(simple_module, analyze_every=4)

        # Count how many times _before_training_step runs analysis
        original_before = model._before_training_step
        call_count = {"value": 0}

        def counting_before(batch, batch_idx):
            if model._should_analyze(model.global_step):
                call_count["value"] += 1
            return original_before(batch, batch_idx)

        model._before_training_step = counting_before

        trainer = pl.Trainer(
            max_steps=8,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, simple_dataloader)

        # With analyze_every=4 and 8 steps, should analyze at steps 0, 4
        assert call_count["value"] == 2

    def test_schedule_reduces_compute(self, simple_module, simple_dataloader):
        """Test schedule reduces analysis calls."""
        from perspic.logger import LogarithmicWindowSchedule

        # Only analyze at steps 0 and 5
        schedule = LogarithmicWindowSchedule(
            windows={0: [0], 1: [5]},
            window_centers={0: 0, 1: 5},
            step_to_window={0: 0, 5: 1},
        )
        model = analyzer(simple_module, analysis_schedule=schedule)

        call_count = {"value": 0}
        original_before = model._before_training_step

        def counting_before(batch, batch_idx):
            if model._should_analyze(model.global_step):
                call_count["value"] += 1
            return original_before(batch, batch_idx)

        model._before_training_step = counting_before

        trainer = pl.Trainer(
            max_steps=8,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, simple_dataloader)

        assert call_count["value"] == 2
