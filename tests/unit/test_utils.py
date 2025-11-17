import pytest
import torch
import torch.nn as nn

from perspic.utils import (
    set_track_running_states,
    save_bn_track_states,
    restore_bn_track_states,
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
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


class TestBatchNormUtils:
    @pytest.mark.parametrize("track", [True, False])
    @pytest.mark.parametrize(
        "model_fixture", ["model_with_bn", "nested_model_with_bn", "model_without_bn"]
    )
    def test_set_track_running_states_and_return(self, request, model_fixture, track):
        # request is a pytest fixture that allows access to other fixtures
        model = request.getfixturevalue(model_fixture)
        # capture a non-BN param (if present) to ensure it's unchanged
        initial_fc1 = model.fc1.weight.clone() if hasattr(model, "fc1") else None

        ret = set_track_running_states(model, track=track)
        assert ret is model

        for m in _bn_modules(model):
            assert m.track_running_stats is track

        if initial_fc1 is not None:
            assert torch.allclose(model.fc1.weight, initial_fc1)

    def test_toggle_multiple_times(self, model_with_bn):
        # True -> False -> True
        set_track_running_states(model_with_bn, track=True)
        set_track_running_states(model_with_bn, track=False)
        set_track_running_states(model_with_bn, track=True)
        for m in _bn_modules(model_with_bn):
            assert m.track_running_stats is True

    @pytest.mark.parametrize(
        "model_fixture, expected_count",
        [("model_with_bn", 2), ("nested_model_with_bn", 2), ("model_without_bn", 0)],
    )
    def test_save_bn_track_states(self, request, model_fixture, expected_count):
        model = request.getfixturevalue(model_fixture)
        states = save_bn_track_states(model)
        assert isinstance(states, list)
        assert len(states) == expected_count

        # if there are BN modules, ensure structure and references
        saved_modules = [m for m, _ in states]
        for m, state in states:
            assert isinstance(m, nn.modules.batchnorm._BatchNorm)
            assert isinstance(state, bool)

        if expected_count > 0:
            original_bn = _bn_modules(model)
            assert saved_modules == original_bn  # identity/order preserved

    def test_restore_bn_track_states_roundtrip(self, model_with_bn):
        # Save initial (default True), flip to False, restore
        initial = save_bn_track_states(model_with_bn)
        set_track_running_states(model_with_bn, track=False)
        for m in _bn_modules(model_with_bn):
            assert m.track_running_stats is False

        restore_bn_track_states(initial)
        for m in _bn_modules(model_with_bn):
            assert m.track_running_stats is True

    def test_restore_mixed_states(self, model_with_bn):
        set_track_running_states(model_with_bn, track=True)
        bn = _bn_modules(model_with_bn)
        if not bn:
            return
        bn[0].track_running_stats = False  # make mixed
        mixed = save_bn_track_states(model_with_bn)

        # change and restore
        set_track_running_states(model_with_bn, track=True)
        restore_bn_track_states(mixed)

        bn_after = _bn_modules(model_with_bn)
        assert bn_after[0].track_running_stats is False
        for m in bn_after[1:]:
            assert m.track_running_stats is True

    def test_restore_empty_list_noop(self, model_with_bn):
        restore_bn_track_states([])  # should not raise

    def test_workflow_with_forward_pass(self, model_with_bn, sample_batch):
        initial_states = save_bn_track_states(model_with_bn)
        model_with_bn.train()

        initial_rm = model_with_bn.bn1.running_mean.clone()

        set_track_running_states(model_with_bn, track=True)
        _ = model_with_bn(sample_batch)
        tracked_rm = model_with_bn.bn1.running_mean.clone()

        set_track_running_states(model_with_bn, track=False)
        _ = model_with_bn(sample_batch)
        untracked_rm = model_with_bn.bn1.running_mean.clone()

        assert not torch.allclose(initial_rm, tracked_rm)
        assert torch.allclose(tracked_rm, untracked_rm)

        restore_bn_track_states(initial_states)
        for m in _bn_modules(model_with_bn):
            assert m.track_running_stats is True

    def test_multiple_save_restore_cycles(self, model_with_bn):
        for _ in range(3):
            states = save_bn_track_states(model_with_bn)
            set_track_running_states(model_with_bn, track=False)
            for m in _bn_modules(model_with_bn):
                assert m.track_running_stats is False
            restore_bn_track_states(states)
            for m in _bn_modules(model_with_bn):
                assert m.track_running_stats is True
