import pytest
import torch
import torch.nn as nn

from perspic.utils import (
    set_track_running_stats,
)


class SimpleModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return self.fc3(x)


class ModelWithoutBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class NestedModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(10, 20), nn.BatchNorm1d(20), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(20, 10), nn.BatchNorm1d(10), nn.ReLU())
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x)


@pytest.fixture
def model_with_bn():
    torch.manual_seed(42)
    return SimpleModelWithBatchNorm()


@pytest.fixture
def model_without_bn():
    torch.manual_seed(42)
    return ModelWithoutBatchNorm()


@pytest.fixture
def nested_model_with_bn():
    torch.manual_seed(42)
    return NestedModelWithBatchNorm()


@pytest.fixture
def sample_batch():
    torch.manual_seed(42)
    return torch.randn(8, 10)


def _bn_modules(model):
    return [
        m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]


class TestBatchNormUtils:
    @pytest.mark.parametrize("track", [True, False])
    @pytest.mark.parametrize(
        "model_fixture", ["model_with_bn", "nested_model_with_bn", "model_without_bn"]
    )
    def test_set_track_running_stats_and_return(self, request, model_fixture, track):
        # request is a pytest fixture that allows access to other fixtures
        model = request.getfixturevalue(model_fixture)
        # capture a non-BN param (if present) to ensure it's unchanged
        initial_fc1 = model.fc1.weight.clone() if hasattr(model, "fc1") else None

        ret = set_track_running_stats(model, track=track)
        assert ret is model

        for m in _bn_modules(model):
            assert m.track_running_stats is track

        if initial_fc1 is not None:
            assert torch.allclose(model.fc1.weight, initial_fc1)

    def test_toggle_multiple_times(self, model_with_bn):
        # True -> False -> True
        set_track_running_stats(model_with_bn, track=True)
        set_track_running_stats(model_with_bn, track=False)
        set_track_running_stats(model_with_bn, track=True)
        for m in _bn_modules(model_with_bn):
            assert m.track_running_stats is True
