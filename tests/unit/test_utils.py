"""Unit tests for the utils module."""

import pytest
import torch
import torch.nn as nn

from examples.models import BatchNormMLP, ResidualMLP, WideResNet
from perspic.utils import BatchStatSnapshot


# Test fixtures
@pytest.fixture
def simple_batchnorm_model():
    """Create a simple model with BatchNorm1d for testing."""
    return nn.Sequential(
        nn.Linear(10, 20), nn.BatchNorm1d(20), nn.ReLU(), nn.Linear(20, 2)
    )


@pytest.fixture
def batchnorm2d_model():
    """Create a simple CNN model with BatchNorm2d for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 10),
    )


@pytest.fixture
def mixed_batchnorm_model():
    """Create a model with multiple BatchNorm types."""

    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.bn2d = nn.BatchNorm2d(8)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(8 * 32 * 32, 20)
            self.bn1d = nn.BatchNorm1d(20)
            self.output = nn.Linear(20, 2)

        def forward(self, x):
            x = self.bn2d(self.conv(x))
            x = self.flatten(x)
            x = self.bn1d(self.fc(x))
            return self.output(x)

    return MixedModel()


@pytest.fixture
def model_without_batchnorm():
    """Create a model without BatchNorm layers."""
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))


@pytest.fixture
def sample_data_1d():
    """Create sample 1D data for testing."""
    torch.manual_seed(42)
    return torch.randn(8, 10)


@pytest.fixture
def sample_data_2d():
    """Create sample 2D (image) data for testing."""
    torch.manual_seed(42)
    return torch.randn(8, 3, 32, 32)


@pytest.fixture
def batchnorm_mlp_model():
    """Create BatchNormMLP model from examples."""
    return BatchNormMLP(width1=128, width2=128)


@pytest.fixture
def residual_mlp_model():
    """Create ResidualMLP model from examples."""
    return ResidualMLP(hidden_size=64, num_blocks=2)


@pytest.fixture
def wideresnet_model():
    """Create WideResNet model from examples."""
    return WideResNet(depth=10, num_classes=10, widen_factor=8, dropRate=0.0)


# Test Classes
class TestBatchStatSnapshot:
    """Test BatchStatSnapshot context manager."""

    @pytest.mark.parametrize(
        "model_fixture,data_fixture,bn_layer_index",
        [
            ("simple_batchnorm_model", "sample_data_1d", 1),
            ("batchnorm2d_model", "sample_data_2d", 1),
            ("mixed_batchnorm_model", "sample_data_2d", None),
        ],
    )
    def test_running_stats_updated_to_batch_stats(
        self, model_fixture, data_fixture, bn_layer_index, request
    ):
        """Test that running stats match current batch stats inside context.

        This test verifies the core functionality: after entering the context,
        the model's eval mode output matches its train mode output with the
        same data. This is achieved by updating running stats to batch stats
        and applying Bessel's correction.

        Parameterized across three scenarios:
        - BatchNorm1d: Tests correction with batch-only statistics
        - BatchNorm2d: Tests correction with spatial dimensions (images)
        - Mixed model: Tests handling of multiple BatchNorm types together
        """
        model = request.getfixturevalue(model_fixture)
        data = request.getfixturevalue(data_fixture)

        # Get output from fresh model in train mode (this is what context should match)
        model.train()
        original_out = model(data)

        # Update model with different data to have non-trivial running stats
        model.train()
        if "2d" in data_fixture:
            _ = model(torch.randn(16, 3, 32, 32))
        else:
            _ = model(torch.randn(16, 10))

        # Get BatchNorm layer (handle different model structures)
        if bn_layer_index is not None:
            bn_layer = model[bn_layer_index]
        else:
            # For mixed model, check both bn1d and bn2d
            bn_layers = [model.bn1d, model.bn2d]

        # Store running stats before using BatchStatSnapshot
        if bn_layer_index is not None:
            pre_snapshot_mean = bn_layer.running_mean.clone()
            pre_snapshot_var = bn_layer.running_var.clone()
        else:
            pre_snapshot_means = [bn.running_mean.clone() for bn in bn_layers]
            pre_snapshot_vars = [bn.running_var.clone() for bn in bn_layers]

        # Use the context manager
        with BatchStatSnapshot(model, data):
            # Inside context, running stats should have been updated
            if bn_layer_index is not None:
                assert not torch.allclose(
                    bn_layer.running_mean, pre_snapshot_mean, atol=1e-5
                )
                assert not torch.allclose(
                    bn_layer.running_var, pre_snapshot_var, atol=1e-5
                )

                # Verify the stats are reasonable (positive variance, finite values)
                assert torch.all(torch.isfinite(bn_layer.running_mean))
                assert torch.all(torch.isfinite(bn_layer.running_var))
                assert torch.all(bn_layer.running_var > 0)
            else:
                # Check all BatchNorm layers in mixed model
                for bn, pre_mean, pre_var in zip(
                    bn_layers, pre_snapshot_means, pre_snapshot_vars
                ):
                    assert not torch.allclose(bn.running_mean, pre_mean, atol=1e-5)
                    assert not torch.allclose(bn.running_var, pre_var, atol=1e-5)
                    assert torch.all(torch.isfinite(bn.running_mean))
                    assert torch.all(torch.isfinite(bn.running_var))
                    assert torch.all(bn.running_var > 0)

            # Verify that the model output matches train mode with same data
            context_out = model(data)
            assert torch.allclose(context_out, original_out, atol=1e-5)

    def test_momentum_set_to_one_inside_context(
        self, simple_batchnorm_model, sample_data_1d
    ):
        """Test that momentum is set to 1.0 inside context."""
        model = simple_batchnorm_model
        data = sample_data_1d

        bn_layer = model[1]
        original_momentum = bn_layer.momentum

        with BatchStatSnapshot(model, data):
            assert bn_layer.momentum == 1.0

        # Momentum should be restored
        assert bn_layer.momentum == original_momentum

    def test_model_in_eval_mode_inside_context(
        self, simple_batchnorm_model, sample_data_1d
    ):
        """Test that model is in eval mode inside context."""
        model = simple_batchnorm_model.train()
        data = sample_data_1d

        with BatchStatSnapshot(model, data):
            assert not model.training
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    assert not module.training

    def test_train_mode_restored_after_exit(
        self, simple_batchnorm_model, sample_data_1d
    ):
        """Test that train mode is restored after exiting context."""
        model = simple_batchnorm_model.train()
        data = sample_data_1d

        with BatchStatSnapshot(model, data):
            pass

        assert model.training
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                assert module.training

    def test_eval_mode_restored_after_exit(
        self, simple_batchnorm_model, sample_data_1d
    ):
        """Test that eval mode is restored after exiting context."""
        model = simple_batchnorm_model.eval()
        data = sample_data_1d

        with BatchStatSnapshot(model, data):
            pass

        assert not model.training
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                assert not module.training

    def test_momentum_restored_after_exit(self, simple_batchnorm_model, sample_data_1d):
        """Test that original momentum is restored after exiting context."""
        model = simple_batchnorm_model
        data = sample_data_1d

        bn_layer = model[1]
        original_momentum = 0.2
        bn_layer.momentum = original_momentum

        with BatchStatSnapshot(model, data):
            assert bn_layer.momentum == 1.0

        assert bn_layer.momentum == original_momentum

    def test_works_with_batchnorm1d(self, simple_batchnorm_model, sample_data_1d):
        """Test that context manager works with BatchNorm1d layers."""
        model = simple_batchnorm_model
        data = sample_data_1d

        with BatchStatSnapshot(model, data):
            bn_layer = model[1]
            assert isinstance(bn_layer, nn.BatchNorm1d)
            assert not bn_layer.training
            assert bn_layer.momentum == 1.0

    def test_works_with_batchnorm2d(self, batchnorm2d_model, sample_data_2d):
        """Test that context manager works with BatchNorm2d layers."""
        model = batchnorm2d_model
        data = sample_data_2d

        with BatchStatSnapshot(model, data):
            bn_layer = model[1]
            assert isinstance(bn_layer, nn.BatchNorm2d)
            assert not bn_layer.training
            assert bn_layer.momentum == 1.0

    def test_works_with_mixed_batchnorm_types(
        self, mixed_batchnorm_model, sample_data_2d
    ):
        """Test that context manager works with multiple BatchNorm types."""
        model = mixed_batchnorm_model
        data = sample_data_2d

        with BatchStatSnapshot(model, data):
            assert not model.bn1d.training
            assert not model.bn2d.training
            assert model.bn1d.momentum == 1.0
            assert model.bn2d.momentum == 1.0

    def test_no_op_when_no_batchnorm(self, model_without_batchnorm, sample_data_1d):
        """Test that context manager is a no-op for models without BatchNorm.

        Ensures the context manager handles models without BatchNorm layers
        gracefully, leaving them unchanged.
        """
        model = model_without_batchnorm.train()
        data = sample_data_1d

        # Should not raise any errors
        with BatchStatSnapshot(model, data):
            pass

        # Model should still be in train mode (unchanged)
        assert model.training

    def test_state_restored_on_exception(self, simple_batchnorm_model, sample_data_1d):
        """Test that state is restored even when exception is raised."""
        model = simple_batchnorm_model.train()
        data = sample_data_1d

        bn_layer = model[1]
        original_momentum = 0.15
        bn_layer.momentum = original_momentum

        with pytest.raises(RuntimeError, match="Test exception"):
            with BatchStatSnapshot(model, data):
                raise RuntimeError("Test exception")

        # State should be restored despite exception
        assert model.training
        assert bn_layer.training
        assert bn_layer.momentum == original_momentum

    def test_inplace_modification(self, simple_batchnorm_model, sample_data_1d):
        """Test that model is modified in-place.

        Verifies that the context manager modifies the original model object
        rather than creating a copy. The model and the value returned by
        __enter__ should be the same object (same id).
        """
        model = simple_batchnorm_model
        data = sample_data_1d

        with BatchStatSnapshot(model, data) as model_snapshot:
            # Both should point to the same object
            assert model is model_snapshot
            assert id(model) == id(model_snapshot)

    def test_multiple_sequential_uses(self, simple_batchnorm_model, sample_data_1d):
        """Test multiple sequential uses of the context manager."""
        model = simple_batchnorm_model.train()
        data1 = sample_data_1d
        data2 = torch.randn(8, 10)

        bn_layer = model[1]
        original_momentum = bn_layer.momentum

        # First use
        with BatchStatSnapshot(model, data1):
            assert bn_layer.momentum == 1.0

        assert bn_layer.momentum == original_momentum
        assert model.training

        # Second use
        with BatchStatSnapshot(model, data2):
            assert bn_layer.momentum == 1.0

        assert bn_layer.momentum == original_momentum
        assert model.training

    def test_preserves_custom_momentum_values(
        self, simple_batchnorm_model, sample_data_1d
    ):
        """Test that custom momentum values are preserved."""
        model = simple_batchnorm_model
        data = sample_data_1d

        bn_layer = model[1]
        custom_momentum = 0.05
        bn_layer.momentum = custom_momentum

        with BatchStatSnapshot(model, data):
            pass

        assert bn_layer.momentum == custom_momentum

    def test_context_manager_returns_model(
        self, simple_batchnorm_model, sample_data_1d
    ):
        """Test that __enter__ returns the model."""
        model = simple_batchnorm_model
        data = sample_data_1d

        with BatchStatSnapshot(model, data) as returned_model:
            assert returned_model is model

    @pytest.mark.parametrize(
        "model_fixture,data_fixture",
        [
            ("batchnorm_mlp_model", "sample_data_2d"),
            ("residual_mlp_model", "sample_data_2d"),
            ("wideresnet_model", "sample_data_2d"),
        ],
    )
    def test_production_models(self, model_fixture, data_fixture, request):
        """Test that context manager works with production models.

        This test verifies that BatchStatSnapshot produces identical outputs
        to train mode for real-world models used in production/examples:
        - BatchNormMLP: MLP with BatchNorm1d layers
        - ResidualMLP: Residual blocks with BatchNorm1d
        - WideResNet: CNN with BatchNorm2d in residual blocks

        The test ensures that after entering the context, the model's eval
        mode output matches its train mode output with the same data.
        """
        model = request.getfixturevalue(model_fixture)
        data = request.getfixturevalue(data_fixture)

        # Get output from fresh model in train mode (ground truth)
        model.train()
        train_out = model(data)

        # Update model with different data to create non-trivial running stats
        model.train()
        _ = model(torch.randn(16, 3, 32, 32))

        # Use BatchStatSnapshot and verify outputs match train mode
        with BatchStatSnapshot(model, data):
            # Model should be in eval mode inside context
            assert not model.training

            # Get output inside context
            context_out = model(data)

            # Verify outputs match train mode (within tolerance)
            assert torch.allclose(context_out, train_out, atol=1e-5), (
                f"Model {model_fixture}: context output doesn't match train output. "
                f"Max diff: {(context_out - train_out).abs().max():.2e}"
            )

        # Verify model is back in train mode after exit
        assert model.training

    def test_running_stats_restored_after_exit(
        self, simple_batchnorm_model, sample_data_1d
    ):
        """Test that running stats are restored after exiting context."""
        model = simple_batchnorm_model
        data = sample_data_1d

        bn_layer = model[1]

        # Set some initial running stats
        initial_mean = torch.randn_like(bn_layer.running_mean)
        initial_var = torch.rand_like(bn_layer.running_var)
        bn_layer.running_mean.copy_(initial_mean)
        bn_layer.running_var.copy_(initial_var)

        with BatchStatSnapshot(model, data):
            # Stats should change inside context
            assert not torch.allclose(bn_layer.running_mean, initial_mean)
            assert not torch.allclose(bn_layer.running_var, initial_var)

        # Stats should be restored after exit
        assert torch.allclose(
            bn_layer.running_mean, initial_mean
        ), "Running mean not restored"
        assert torch.allclose(
            bn_layer.running_var, initial_var
        ), "Running var not restored"
