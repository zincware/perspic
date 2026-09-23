"""Tests for shared SamplewiseCalculator base class functionality."""

import warnings

import pytest
import torch
import torch.nn as nn

from perspic.calculator.samplewise import SamplewiseCalculator
from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus
from perspic.utils import BatchStatSnapshot


class BatchNormMLP(nn.Module):
    """MLP with BatchNorm for testing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleMLP(nn.Module):
    """MLP without BatchNorm for testing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        return self.layers(x)


class TestBatchNormTrainingWarning:
    """Test that both calculators warn when BatchNorm is in training mode."""

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_warning_when_batchnorm_in_training_mode_network(self, calculator):
        """Verify warning is raised for network gradient computation."""
        torch.manual_seed(42)
        model = BatchNormMLP()
        model.train()
        X = torch.randn(8, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # The computation will fail with BatchNorm in training mode,
            # but the warning should be issued before the failure
            try:
                calculator._compute_per_sample_gradient_norm_network(model, X)
            except (ValueError, RuntimeError):
                # Expected - BatchNorm in training mode fails with vmap/sample-wise ops
                pass

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 1
            assert issubclass(bn_warnings[0].category, UserWarning)
            assert "BatchStatSnapshot" in str(bn_warnings[0].message)

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_warning_when_batchnorm_in_training_mode_loss(self, calculator):
        """Verify warning is raised for loss gradient computation."""
        torch.manual_seed(42)
        model = BatchNormMLP()
        model.train()
        X = torch.randn(8, 10)
        y = torch.randint(0, 2, (8,))
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculator._compute_per_sample_gradient_norm_loss(model, loss_fn, X, y)

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 1
            assert issubclass(bn_warnings[0].category, UserWarning)
            assert "BatchStatSnapshot" in str(bn_warnings[0].message)

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_no_warning_with_batch_stat_snapshot(self, calculator):
        """Verify no warning when BatchStatSnapshot is used."""
        torch.manual_seed(42)
        model = BatchNormMLP()
        X = torch.randn(8, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with BatchStatSnapshot(model, X):
                calculator._compute_per_sample_gradient_norm_network(model, X)

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 0

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_no_warning_without_batchnorm(self, calculator):
        """Verify no warning when model has no BatchNorm layers."""
        torch.manual_seed(42)
        model = SimpleMLP()
        model.train()
        X = torch.randn(8, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculator._compute_per_sample_gradient_norm_network(model, X)

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 0

    @pytest.mark.parametrize(
        "calculator",
        [
            SamplewiseCalculatorFunctorch,
            SamplewiseCalculatorOpacus,
        ],
    )
    def test_no_warning_when_batchnorm_in_eval_mode(self, calculator):
        """Verify no warning when BatchNorm is in eval mode (even without snapshot)."""
        torch.manual_seed(42)
        model = BatchNormMLP()
        model.eval()  # Explicitly set to eval mode
        X = torch.randn(8, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculator._compute_per_sample_gradient_norm_network(model, X)

            bn_warnings = [x for x in w if "BatchNorm" in str(x.message)]
            assert len(bn_warnings) == 0


class TestResolveTargetMask:
    """Tests for `SamplewiseCalculator.resolve_target_mask`, the single place
    both calculator backends decide whether/how to mask -- they must never
    disagree, since `compute()` uses the same `n_elements` to normalize both
    `batch_grad_norms_network` and `batch_grad_norms_loss`, and only that
    agreement is what makes CouplingCalculator's chi_pos independent of
    `n_elements` (see `coupling.py` and `TestCouplingCancellation` below).
    """

    def test_ignore_index_none_disables_masking(self):
        """Passing ignore_index=None must skip masking even if -100 is
        present in targets -- the documented escape hatch for data where
        -100 is a legitimate label."""
        targets = torch.tensor([-100, 1, 2])
        mask, n_elements = SamplewiseCalculator.resolve_target_mask(
            targets, ignore_index=None
        )
        assert mask is None
        assert n_elements == targets.numel()

    def test_no_ignore_index_present_returns_none(self):
        """A batch that happens to contain no -100 must resolve to mask=None
        (not an all-True tensor) -- this is what lets callers skip the
        masking multiply entirely and stay bitwise identical to before this
        feature existed."""
        targets = torch.tensor([[0, 1, 2], [3, 0, 1]])
        mask, n_elements = SamplewiseCalculator.resolve_target_mask(
            targets, ignore_index=-100
        )
        assert mask is None
        assert n_elements == targets.numel()

    def test_masks_ignore_index_positions(self):
        """A batch containing -100 must produce a boolean mask (True = real
        position) shaped like targets, and n_elements = number of real
        positions, not targets.numel()."""
        targets = torch.tensor([[5, -100, 2, -100], [1, 2, 3, 4]])
        mask, n_elements = SamplewiseCalculator.resolve_target_mask(
            targets, ignore_index=-100
        )
        assert mask is not None
        assert mask.dtype == torch.bool
        assert mask.shape == targets.shape
        expected = torch.tensor([[True, False, True, False], [True, True, True, True]])
        assert torch.equal(mask, expected)
        assert n_elements == expected.sum()
        assert n_elements.item() == 6  # 8 positions total, 2 masked out

    def test_float_targets_never_masked(self):
        """One-hot / soft (float) targets must never be masked, even if a
        value happens to equal -100.0 -- compute()'s normalize=False is the
        documented path for those, per the existing docstring."""
        targets = torch.tensor([[-100.0, 1.0], [0.3, 0.7]])
        mask, n_elements = SamplewiseCalculator.resolve_target_mask(
            targets, ignore_index=-100
        )
        assert mask is None
        assert n_elements == targets.numel()

    def test_custom_ignore_index(self):
        """A non-default ignore_index value must be respected."""
        targets = torch.tensor([0, 1, -1, 2])
        mask, n_elements = SamplewiseCalculator.resolve_target_mask(
            targets, ignore_index=-1
        )
        assert mask is not None
        assert torch.equal(mask, torch.tensor([True, True, False, True]))
        assert n_elements.item() == 3


class TestBroadcastMask:
    """Tests for `SamplewiseCalculator.broadcast_mask`."""

    def test_appends_trailing_singleton_dims(self):
        mask = torch.ones(2, 3, dtype=torch.bool)
        out = SamplewiseCalculator.broadcast_mask(mask, ndim=4)
        assert out.shape == (2, 3, 1, 1)

    def test_noop_when_already_correct_ndim(self):
        mask = torch.ones(2, 3, dtype=torch.bool)
        out = SamplewiseCalculator.broadcast_mask(mask, ndim=2)
        assert out.shape == (2, 3)

    def test_result_broadcasts_against_target_shape(self):
        """The whole point: multiplying against a same-batch, larger tensor
        must work via ordinary broadcasting once reshaped."""
        mask = torch.tensor([[True, False], [True, True]])  # (2, 2)
        out = torch.ones(2, 2, 5)  # e.g. (batch, seq_len, vocab)
        mask_b = SamplewiseCalculator.broadcast_mask(mask, out.dim())
        result = out * mask_b
        assert result.shape == out.shape
        assert torch.all(result[0, 1] == 0)  # masked position zeroed
        assert torch.all(result[0, 0] == 1)  # real position untouched


class TestCouplingCancellation:
    """Verifies the claim that CouplingCalculator's chi_pos (chi_coup) is
    independent of the `n_elements` normalization factor -- chi_loss is
    multiplied by it and chi_net divided by it, so it cancels in their
    product. This is what makes it safe for `resolve_target_mask` to change
    what `n_elements` means (targets.numel() -> count of real positions)
    without perturbing chi_pos through the normalization alone; only the
    *masked* chi_net projection (a different number now, not just rescaled)
    can still move chi_pos.
    """

    def test_chi_coup_invariant_to_normalization_factor(self):
        from perspic.calculator.coupling import CouplingCalculator

        raw_net = 3.7  # chi_net before normalization (an arbitrary example)
        raw_loss = 5.2  # chi_loss before normalization
        delta_loss = -2.0
        calc = CouplingCalculator()

        # Two different normalization factors, e.g. N=20 (targets.numel())
        # vs. M=11 (masked real-position count) -- values chosen to echo
        # the ~44.6% pad fraction measured for SimpleStories (N/M ~= 1.8).
        n_big, n_small = 20.0, 11.0

        coup_n = calc.calculate(
            delta_loss=delta_loss, chi_loss=raw_loss * n_big, chi_net=raw_net / n_big
        )
        coup_m = calc.calculate(
            delta_loss=delta_loss,
            chi_loss=raw_loss * n_small,
            chi_net=raw_net / n_small,
        )
        assert coup_n == pytest.approx(coup_m, rel=1e-12)
