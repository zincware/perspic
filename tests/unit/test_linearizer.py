import pickle

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


class TestMemoryUsageOfSaveLoad:
    """Test for memory usage overhead of Linearizer."""

    pass


class TestMultipleLearningRates:
    def test_probe_train_step_multiple_etas(self, simple_model):
        """Test that probe_train_step can handle multiple learning rates."""
        torch.manual_seed(42)
        model = simple_model
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))
        eta_array = [1e-1, 1e-2, 1e-3, 1e-6]
        linearizer = Linearizer(eta_array=eta_array)

        results = linearizer.probe_train_step(
            model=model,
            criterion=criterion,
            x=x,
            y=y,
            scheduler=None,
        )

        assert results is not None
        assert isinstance(results, dict)
        assert len(results) == len(eta_array)

        # Verify all etas are in results with valid values
        for eta, (loss, perturbed_loss) in results.items():
            assert eta in eta_array
            assert isinstance(loss, float)
            if perturbed_loss is not None:
                assert isinstance(perturbed_loss, float)

        # Verify not all perturbed losses are identical
        perturbed_losses = [
            results[eta][1] for eta in eta_array if results[eta][1] is not None
        ]
        unperturbed_losses = [results[eta][0] for eta in eta_array]

        unique_losses = len(set(perturbed_losses))
        assert unique_losses > 1, (
            f"Expected at least 2 different loss values, "
            f"got {unique_losses}: {perturbed_losses}"
        )
        assert len(unperturbed_losses) == len(eta_array)
        assert all(
            loss == unperturbed_losses[0] for loss in unperturbed_losses
        ), "Unperturbed losses differ: {unperturbed_losses}"


class TestLoadModelState:
    """Test for _load_model_state function. Should only test for loading from bytes,
    rejecting non-bytes and accepting optional device parameter. For full save/load
    integration tests, see TestSaveLoadIntegration below.
    """

    def test_load_accepts_bytes_state(self, simple_model, device):
        simple_model = simple_model.to(device)
        saved_bytes = Linearizer._save_model_state(simple_model)
        assert isinstance(saved_bytes, bytes)

        result = Linearizer._load_model_state(simple_model, saved_bytes, device=device)
        assert isinstance(result, torch.nn.Module)
        assert result is simple_model

    def test_load_rejects_non_bytes_state(self, simple_model, device):
        simple_model = simple_model.to(device)
        invalid_state = "not bytes"
        with pytest.raises((TypeError, pickle.UnpicklingError, EOFError)):
            Linearizer._load_model_state(simple_model, invalid_state)

    def test_load_rejects_dict_state(self, simple_model, device):
        simple_model = simple_model.to(device)
        state_dict = simple_model.state_dict()
        with pytest.raises((TypeError, pickle.UnpicklingError)):
            Linearizer._load_model_state(simple_model, state_dict)

    def test_load_empty_model_state(self, simple_model, device):
        simple_model = simple_model.to(device)
        empty_state = b""
        with pytest.raises((KeyError, EOFError)):
            Linearizer._load_model_state(simple_model, empty_state)

    def test_load_accepts_optional_device(self, simple_model, device):
        simple_model = simple_model.to(device)
        saved_bytes = Linearizer._save_model_state(simple_model)

        # Should accept None
        result1 = Linearizer._load_model_state(simple_model, saved_bytes, device=None)
        assert isinstance(result1, torch.nn.Module)

        # Should accept torch.device
        result2 = Linearizer._load_model_state(simple_model, saved_bytes, device=device)
        assert isinstance(result2, torch.nn.Module)

    def test_load_rejects_invalid_device_type(self, simple_model, device):
        simple_model = simple_model.to(device)
        saved_bytes = Linearizer._save_model_state(simple_model)

        # Should raise or handle gracefully
        with pytest.raises((TypeError, RuntimeError)):
            Linearizer._load_model_state(simple_model, saved_bytes, device="invalid")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_moves_tensors_to_gpu(self, simple_model):
        """Test that load can move model to GPU."""
        simple_model = simple_model.cuda()
        saved_bytes = Linearizer._save_model_state(simple_model)

        # Load to GPU
        device = torch.device("cuda")
        Linearizer._load_model_state(simple_model, saved_bytes, device=device)

        # Verify all params are on GPU
        for param in simple_model.parameters():
            assert (
                param.device.type == "cuda"
            ), f"Parameter should be on CUDA, got {param.device}"


class TestSaveModelState:
    """Test for _save_model_state function."""

    def test_save_returns_bytes(self, simple_model, device):
        simple_model = simple_model.to(device)
        state_bytes = Linearizer._save_model_state(simple_model)
        assert isinstance(state_bytes, bytes)

    def test_save_non_empty_bytes(self, simple_model, device):
        simple_model = simple_model.to(device)
        state_bytes = Linearizer._save_model_state(simple_model)
        assert len(state_bytes) > 0

    def test_save_consistent_across_calls(self, simple_model, device):
        simple_model = simple_model.to(device)
        saved_bytes_1 = Linearizer._save_model_state(simple_model)
        saved_bytes_2 = Linearizer._save_model_state(simple_model)
        assert saved_bytes_1 == saved_bytes_2

    def test_save_two_different_models(self, simple_model, complex_model, device):
        simple_model = simple_model.to(device)
        complex_model = complex_model.to(device)
        saved_bytes_1 = Linearizer._save_model_state(simple_model)
        saved_bytes_2 = Linearizer._save_model_state(complex_model)
        assert saved_bytes_1 != saved_bytes_2

    def test_save_differs_when_parameters_change(self, simple_model, device):
        simple_model = simple_model.to(device)
        saved_bytes_1 = Linearizer._save_model_state(simple_model)
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(999.0)
                break
        saved_bytes_2 = Linearizer._save_model_state(simple_model)
        assert saved_bytes_1 != saved_bytes_2


class TestProbeTrainStepCore:
    """Test core probing algorithm behavior."""

    def test_gradients_are_zeroed_after_probing(self, simple_model):
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        linearizer = Linearizer([1e-3])
        linearizer.probe_train_step(model=simple_model, criterion=criterion, x=x, y=y)

        # All gradients should be None or zero
        for param in simple_model.parameters():
            if param.grad is not None:
                assert torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                ), "Gradients should be zeroed after probing"

    def test_exception_in_probe_stores_none(self, simple_model):
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        # Mock criterion to fail on second call (perturbed loss)
        original_criterion = criterion
        call_count = [0]

        def failing_criterion(pred, target):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call (perturbed)
                raise RuntimeError("Mock error")
            return original_criterion(pred, target)

        linearizer = Linearizer([1e-3])
        results = linearizer.probe_train_step(
            model=simple_model,
            criterion=failing_criterion,
            x=x,
            y=y,
        )

        # Should have original loss but None for perturbed
        assert results[1e-3][0] is not None
        assert results[1e-3][1] is None

    def test_original_loss_identical_across_etas(self, simple_model):
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))
        eta_array = [1e-1, 1e-3, 1e-5, 1e-7]

        linearizer = Linearizer(eta_array)
        results = linearizer.probe_train_step(
            model=simple_model,
            criterion=criterion,
            x=x,
            y=y,
        )

        original_losses = [results[eta][0] for eta in eta_array]
        assert all(
            loss == original_losses[0] for loss in original_losses
        ), f"Original losses should be identical: {original_losses}"


class TestSaveLoadIntegration:
    def test_save_and_load_round_trip(self, simple_model):
        """Test complete save and load cycle preserves model state."""
        original_state = {k: v.clone() for k, v in simple_model.state_dict().items()}

        saved_bytes = Linearizer._save_model_state(simple_model)

        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(0.0)

        result = Linearizer._load_model_state(simple_model, saved_bytes)

        assert result is simple_model
        for key in original_state:
            assert torch.allclose(
                simple_model.state_dict()[key], original_state[key]
            ), f"Parameter {key} not restored after round-trip"

    def test_multiple_save_load_cycles(self, simple_model):
        """Test multiple consecutive save/load cycles."""
        original_state = {k: v.clone() for k, v in simple_model.state_dict().items()}

        for _ in range(3):
            saved_bytes = Linearizer._save_model_state(simple_model)
            with torch.no_grad():
                for param in simple_model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            Linearizer._load_model_state(simple_model, saved_bytes)

        for key in original_state:
            assert torch.allclose(
                simple_model.state_dict()[key], original_state[key]
            ), f"Parameter {key} not restored after multiple cycles"
