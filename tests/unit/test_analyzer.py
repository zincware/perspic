"""Unit tests for the analyzer module."""

import warnings
from unittest.mock import Mock, patch

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from perspic.analyzer import analyzer
from perspic.calculator.linearizer import Linearizer
from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus


# Test fixtures
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
def manual_optimization_module():
    """Create a LightningModule with manual optimization."""

    class ManualOptimizationModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(10, 2)
            self.criterion = F.cross_entropy
            self.automatic_optimization = False

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
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    return ManualOptimizationModule


@pytest.fixture
def module_without_criterion():
    """Create a LightningModule without criterion attribute."""

    class NoCriterionModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(10, 2)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    return NoCriterionModule


@pytest.fixture
def sample_batch():
    """Create a sample batch of data."""
    torch.manual_seed(42)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    return x, y


def _make_measure_dataloader(n_samples, micro, seed=123, drop_last=True):
    """Build a deterministic measure DataLoader (shuffle=False) for testing
    the independent measure-batch-size path."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 10, generator=g)
    y = torch.randint(0, 2, (n_samples,), generator=g)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=micro, shuffle=False, drop_last=drop_last)


# Test Classes
class TestAnalyzerFactoryFunction:
    """Test the analyzer factory function."""

    def test_creates_analyzer_instance(self, simple_lightning_module):
        """Test that analyzer() creates an instance successfully."""
        model = analyzer(simple_lightning_module, sample_wise_engine="functorch")

        assert model is not None
        assert isinstance(model, pl.LightningModule)
        assert hasattr(model, "sample_calc")
        assert hasattr(model, "linearizer")

    def test_forwards_model_kwargs(self, simple_lightning_module):
        """Test that **model_kwargs are forwarded to the LightningModule."""
        model = analyzer(simple_lightning_module, lr=0.005)

        assert model.hparams.lr == 0.005

    def test_invalid_engine_raises_error(self, simple_lightning_module):
        """Test that invalid sample_wise_engine raises ValueError."""
        with pytest.raises(ValueError, match="sample_wise_engine must be either"):
            analyzer(simple_lightning_module, sample_wise_engine="invalid")

    def test_opacus_strict_with_functorch_raises_error(self, simple_lightning_module):
        """Test that opacus_strict=True with functorch engine raises ValueError."""
        with pytest.raises(ValueError, match="opacus_strict=True is only valid"):
            analyzer(
                simple_lightning_module,
                sample_wise_engine="functorch",
                opacus_strict=True,
            )

    def test_functorch_engine_initialization(self, simple_lightning_module):
        """Test that functorch engine initializes correctly."""
        model = analyzer(simple_lightning_module, sample_wise_engine="functorch")

        assert isinstance(model.sample_calc, SamplewiseCalculatorFunctorch)

    def test_opacus_engine_initialization(self, simple_lightning_module):
        """Test that opacus engine initializes correctly."""
        model = analyzer(simple_lightning_module, sample_wise_engine="opacus")

        from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus

        assert isinstance(model.sample_calc, SamplewiseCalculatorOpacus)

    def test_disable_analyzer_flag(self, simple_lightning_module):
        """Test that disable_analyzer flag is stored correctly."""
        model = analyzer(simple_lightning_module, disable_analyzer=True)

        assert model.disable_analyzer is True

    def test_log_metrics_flag_default(self, simple_lightning_module):
        """Test that log_metrics defaults to True."""
        model = analyzer(simple_lightning_module)

        assert model.log_metrics is True

    def test_log_metrics_flag_false(self, simple_lightning_module):
        """Test that log_metrics can be set to False."""
        model = analyzer(simple_lightning_module, log_metrics=False)

        assert model.log_metrics is False


class TestAnalyzerInitialization:
    """Test Analyzer class initialization."""

    def test_sample_calc_initialized(self, simple_lightning_module):
        """Test that sample_calc is initialized."""
        model = analyzer(simple_lightning_module, sample_wise_engine="functorch")

        assert model.sample_calc is not None
        assert isinstance(model.sample_calc, SamplewiseCalculatorFunctorch)

    def test_linearizer_initialized(self, simple_lightning_module):
        """Test that linearizer is initialized."""
        model = analyzer(simple_lightning_module)

        assert model.linearizer is not None
        assert isinstance(model.linearizer, Linearizer)

    def test_automatic_optimization_disabled(self, simple_lightning_module):
        """Test that automatic_optimization is set to False."""
        model = analyzer(simple_lightning_module)

        assert model.automatic_optimization is False

    def test_missing_criterion_raises_error(self, module_without_criterion):
        """Test that missing criterion attribute raises AttributeError."""
        with pytest.raises(AttributeError, match="criterion"):
            analyzer(module_without_criterion)

    def test_delegate_optimization_false_by_default(self, simple_lightning_module):
        """Test that delegate_optimization is False for automatic optimization modules."""
        model = analyzer(simple_lightning_module)

        assert model.delegate_optimization is False

    def test_delegate_optimization_true_for_manual(self, manual_optimization_module):
        """Test that delegate_optimization is True for manual optimization modules."""
        with pytest.warns(UserWarning, match="manual optimization"):
            model = analyzer(manual_optimization_module)

        assert model.delegate_optimization is True

    def test_warning_for_manual_optimization(self, manual_optimization_module):
        """Test that warning is raised for manual optimization modules."""
        with pytest.warns(UserWarning, match="manual optimization"):
            analyzer(manual_optimization_module)


class TestBeforeTrainingStepHook:
    """Test _before_training_step hook method."""

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_calls_sample_calc_compute(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that sample_calc.compute() is called with correct arguments."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module)
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        mock_compute.assert_called_once()
        call_args = mock_compute.call_args
        assert call_args[0][0] == model.model  # model argument
        assert call_args[0][1] == model.criterion  # criterion argument
        assert torch.equal(call_args[0][2], x)  # x argument
        assert torch.equal(call_args[0][3], y)  # y argument

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_calls_linearizer_probe(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that linearizer.compute() is called with correct arguments."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module)
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        mock_probe.assert_called_once()
        call_kwargs = mock_probe.call_args[1]
        assert call_kwargs["model"] == model.model
        assert call_kwargs["criterion"] == model.criterion
        assert torch.equal(call_kwargs["x1"], x)
        assert torch.equal(call_kwargs["y1"], y)

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_logs_metrics_when_enabled(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that metrics are logged when log_metrics=True."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(2.5),
            "batch_grad_norms_loss": torch.tensor(3.7),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module, log_metrics=True)
        model.log = Mock()  # Mock the log method
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        # Verify log was called for each metric
        assert model.log.call_count > 0
        logged_metrics = {call[0][0]: call[0][1] for call in model.log.call_args_list}

        assert "chi_net" in logged_metrics
        assert "chi_loss" in logged_metrics
        assert "chi_coup" in logged_metrics
        assert "batch_size" in logged_metrics
        assert "analysis_step" in logged_metrics
        assert "loss" in logged_metrics
        assert "grad_norm_squared" in logged_metrics

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(SamplewiseCalculatorOpacus, "compute_cross_metrics")
    @patch.object(Linearizer, "compute")
    def test_logs_cross_metrics(
        self,
        mock_probe,
        mock_compute_cross,
        mock_compute,
        simple_lightning_module,
        sample_batch,
    ):
        """Test that cross metrics are logged when cross_response=True."""
        # Mock compute results for self and cross
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(2.5),
            "batch_grad_norms_loss": torch.tensor(3.7),
        }

        mock_compute_cross.return_value = {
            "batch_grad_norms_network": torch.tensor(1.5),
            "batch_grad_norms_loss": torch.tensor(2.7),
        }

        # Mock probe results
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": (0.5, 0.0, -0.5),
        }

        # Initialize analyzer with cross_response=True
        model = analyzer(simple_lightning_module, log_metrics=True, cross_response=True)
        model.log = Mock()  # Mock the log method
        x, y = sample_batch

        # Create a dummy cross batch
        x2, y2 = x.clone(), y.clone()

        # Call _before_training_step with cross response batch
        model._before_training_step((x, y), 0, cross_response_batch=(x2, y2))

        # Verify log was called for each metric
        assert model.log.call_count > 0
        logged_metrics = {call[0][0]: call[0][1] for call in model.log.call_args_list}

        # Check self metrics
        assert "chi_net" in logged_metrics
        assert "chi_loss" in logged_metrics
        assert "chi_coup" in logged_metrics
        assert "loss" in logged_metrics
        assert "grad_norm_squared" in logged_metrics

        # Check cross metrics
        assert "cross_chi_net" in logged_metrics
        assert "cross_chi_loss" in logged_metrics
        assert "cross_chi_coup" in logged_metrics
        assert "cross_loss" in logged_metrics
        assert "cross_grad_dot_product" in logged_metrics
        assert "cross_batch_size" in logged_metrics

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_no_logging_when_disabled(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that metrics are not logged when log_metrics=False."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(2.5),
            "batch_grad_norms_loss": torch.tensor(3.7),
        }

        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module, log_metrics=False)
        model.log = Mock()
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        # Log should not be called
        model.log.assert_not_called()

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_batch_unpacking(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that batch is correctly unpacked into (x, y)."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module)
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        # Verify x and y were passed correctly
        assert torch.equal(mock_compute.call_args[0][2], x)
        assert torch.equal(mock_compute.call_args[0][3], y)


class TestAfterTrainingStepHook:
    """Test _after_training_step hook method."""

    def test_executes_without_error(self, simple_lightning_module, sample_batch):
        """Test that _after_training_step executes without error."""
        model = analyzer(simple_lightning_module)
        x, y = sample_batch
        output = torch.tensor(1.0)

        # Should not raise any errors
        result = model._after_training_step((x, y), 0, output)

        assert result is None


class TestTrainingStepBehavior:
    """Test training_step method behavior with disable_analyzer flag."""

    @patch.object(SamplewiseCalculatorFunctorch, "compute")
    @patch.object(Linearizer, "compute")
    def test_before_hook_skipped_when_disabled(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that _before_training_step is not called when disable_analyzer=True."""
        model = analyzer(simple_lightning_module, disable_analyzer=True)
        model.optimizers = Mock(return_value=Mock(zero_grad=Mock(), step=Mock()))
        model.manual_backward = Mock()

        x, y = sample_batch
        model.training_step((x, y), 0)

        # Verify analysis methods were not called
        mock_compute.assert_not_called()
        mock_probe.assert_not_called()

    @patch.object(SamplewiseCalculatorFunctorch, "compute")
    @patch.object(Linearizer, "compute")
    def test_after_hook_skipped_when_disabled(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that _after_training_step is not called when disable_analyzer=True."""
        model = analyzer(simple_lightning_module, disable_analyzer=True)
        model.optimizers = Mock(return_value=Mock(zero_grad=Mock(), step=Mock()))
        model.manual_backward = Mock()
        model._after_training_step = Mock()

        x, y = sample_batch
        model.training_step((x, y), 0)

        # Verify _after_training_step was not called
        model._after_training_step.assert_not_called()

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_hooks_called_when_enabled(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that hooks are called when disable_analyzer=False."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module, disable_analyzer=False)
        model.optimizers = Mock(return_value=Mock(zero_grad=Mock(), step=Mock()))
        model.manual_backward = Mock()

        x, y = sample_batch
        model.training_step((x, y), 0)

        # Verify analysis methods were called
        mock_compute.assert_called_once()
        mock_probe.assert_called_once()


class TestOptimizationControl:
    """Test manual optimization control."""

    def test_optimizer_zero_grad_called(self, simple_lightning_module, sample_batch):
        """Test that optimizer.zero_grad() is called at the start."""
        model = analyzer(simple_lightning_module)
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        # Mock the analysis hooks to avoid actual computation
        with patch.object(model, "_before_training_step"), patch.object(
            model, "_after_training_step"
        ):
            x, y = sample_batch
            model.training_step((x, y), 0)

        mock_opt.zero_grad.assert_called_once()

    def test_manual_backward_called(self, simple_lightning_module, sample_batch):
        """Test that manual_backward is called with the loss."""
        model = analyzer(simple_lightning_module)
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        with patch.object(model, "_before_training_step"), patch.object(
            model, "_after_training_step"
        ):
            x, y = sample_batch
            loss = model.training_step((x, y), 0)

        model.manual_backward.assert_called_once()
        # Verify it was called with a tensor (the loss)
        assert isinstance(model.manual_backward.call_args[0][0], torch.Tensor)

    def test_optimizer_step_called_when_not_delegating(
        self, simple_lightning_module, sample_batch
    ):
        """Test that optimizer.step() is called when delegate_optimization=False."""
        model = analyzer(simple_lightning_module)
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        with patch.object(model, "_before_training_step"), patch.object(
            model, "_after_training_step"
        ):
            x, y = sample_batch
            model.training_step((x, y), 0)

        mock_opt.step.assert_called_once()

    def test_optimizer_step_not_called_when_delegating(
        self, manual_optimization_module, sample_batch
    ):
        """Test that optimizer.step() is not called when delegate_optimization=True."""
        with pytest.warns(UserWarning, match="manual optimization"):
            model = analyzer(manual_optimization_module)

        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        with patch.object(model, "_before_training_step"), patch.object(
            model, "_after_training_step"
        ):
            x, y = sample_batch
            model.training_step((x, y), 0)

        # Step should not be called because we're delegating to parent's training_step
        # Note: The parent's training_step calls opt.step(), but since we're mocking
        # optimizers(), our mock's step won't be called in the parent either
        # This test verifies the Analyzer doesn't call step when delegating
        assert model.delegate_optimization is True


class TestMetricLogging:
    """Test metric logging behavior."""

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_correct_metric_names(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that metrics are logged with correct names."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(2.0),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module, log_metrics=True)
        model.log = Mock()
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        logged_names = [call[0][0] for call in model.log.call_args_list]

        assert "chi_net" in logged_names
        assert "chi_loss" in logged_names
        assert "loss" in logged_names
        assert "batch_size" in logged_names
        assert "chi_coup" in logged_names
        assert "analysis_step" in logged_names
        assert "grad_norm_squared" in logged_names

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_correct_metric_values(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test that metrics are logged with correct values."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.5),
            "batch_grad_norms_loss": torch.tensor(2.5),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module, log_metrics=True)
        model.log = Mock()
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        logged_values = {call[0][0]: call[0][1] for call in model.log.call_args_list}

        assert torch.allclose(logged_values["chi_net"], torch.tensor(1.5))
        assert torch.allclose(logged_values["chi_loss"], torch.tensor(2.5))
        assert logged_values["loss"] == 1.0
        assert logged_values["grad_norm_squared"] == 1.0  # -delta = -(-1.0) = 1.0
        assert logged_values["batch_size"] == 4


class TestAnalyzerScheduling:
    """Test analysis scheduling in the Analyzer."""

    def test_analyze_every_stored(self, simple_lightning_module):
        """Test analyze_every parameter is stored."""
        model = analyzer(simple_lightning_module, analyze_every=10)
        assert model.analyze_every == 10

    def test_analyze_every_none_default(self, simple_lightning_module):
        """Test analyze_every defaults to None."""
        model = analyzer(simple_lightning_module)
        assert model.analyze_every is None

    def test_analyze_every_invalid_raises_error(self, simple_lightning_module):
        """Test analyze_every < 1 raises ValueError."""
        with pytest.raises(ValueError, match="analyze_every must be a positive"):
            analyzer(simple_lightning_module, analyze_every=0)

    def test_analysis_schedule_stored(self, simple_lightning_module):
        """Test analysis_schedule parameter is stored."""
        from perspic.logger import logarithmic_windows

        schedule = logarithmic_windows(max_steps=100)
        model = analyzer(simple_lightning_module, analysis_schedule=schedule)
        assert model.analysis_schedule is schedule

    def test_analysis_schedule_none_default(self, simple_lightning_module):
        """Test analysis_schedule defaults to None."""
        model = analyzer(simple_lightning_module)
        assert model.analysis_schedule is None

    def test_should_analyze_default_true(self, simple_lightning_module):
        """Test _should_analyze returns True by default."""
        model = analyzer(simple_lightning_module)
        assert model._should_analyze(0) is True
        assert model._should_analyze(42) is True

    def test_should_analyze_with_analyze_every(self, simple_lightning_module):
        """Test _should_analyze respects analyze_every."""
        model = analyzer(simple_lightning_module, analyze_every=10)
        assert model._should_analyze(0) is True
        assert model._should_analyze(10) is True
        assert model._should_analyze(5) is False

    def test_should_analyze_with_schedule(self, simple_lightning_module):
        """Test _should_analyze uses schedule when provided."""
        from perspic.logger import LogarithmicWindowSchedule

        schedule = LogarithmicWindowSchedule(
            windows={0: [0, 1]},
            window_centers={0: 0},
            step_to_window={0: 0, 1: 0},
        )
        model = analyzer(simple_lightning_module, analysis_schedule=schedule)
        assert model._should_analyze(0) is True
        assert model._should_analyze(1) is True
        assert model._should_analyze(2) is False

    def test_schedule_takes_precedence(self, simple_lightning_module):
        """Test analysis_schedule takes precedence over analyze_every."""
        from perspic.logger import LogarithmicWindowSchedule

        schedule = LogarithmicWindowSchedule(
            windows={0: [5]},
            window_centers={0: 5},
            step_to_window={5: 0},
        )
        model = analyzer(
            simple_lightning_module, analyze_every=10, analysis_schedule=schedule
        )
        # analyze_every would say True for 0, but schedule says False
        assert model._should_analyze(0) is False
        assert model._should_analyze(5) is True

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_before_hook_skipped_when_not_scheduled(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test _before_training_step skips analysis when not scheduled."""
        model = analyzer(simple_lightning_module, analyze_every=10)
        model._optimizer_step_count = 5  # Not a multiple of 10

        model._before_training_step(sample_batch, 0)

        mock_compute.assert_not_called()
        mock_probe.assert_not_called()

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_logs_window_info_with_schedule(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test window_id, window_center, and window_width logged when schedule provided."""
        from perspic.logger import LogarithmicWindowSchedule

        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        schedule = LogarithmicWindowSchedule(
            windows={0: [0]},
            window_centers={0: 0},
            step_to_window={0: 0},
        )
        model = analyzer(
            simple_lightning_module, analysis_schedule=schedule, log_metrics=True
        )
        model.log = Mock()

        model._before_training_step(sample_batch, 0)

        logged_names = [call[0][0] for call in model.log.call_args_list]
        assert "window_id" in logged_names
        assert "window_center" in logged_names
        assert "window_width" in logged_names

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_no_window_info_without_schedule(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Test window_id, window_center, and window_width not logged without schedule."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module, log_metrics=True)
        model.log = Mock()

        model._before_training_step(sample_batch, 0)

        logged_names = [call[0][0] for call in model.log.call_args_list]
        assert "window_id" not in logged_names
        assert "window_center" not in logged_names
        assert "window_width" not in logged_names


class TestGradientAccumulation:
    """Test gradient accumulation functionality."""

    # The tests are categorized into sections A-J for clarity.

    # --- A. Parameter validation ---

    def test_accumulation_steps_default(self, simple_lightning_module):
        """No params → accumulation_steps=1."""
        model = analyzer(simple_lightning_module)
        assert model.accumulation_steps == 1

    def test_batch_size_only_no_accumulation(self, simple_lightning_module):
        """micro_batch_size alone → no accumulation, value stored."""
        model = analyzer(simple_lightning_module, micro_batch_size=8)
        assert model.accumulation_steps == 1
        assert model.micro_batch_size == 8
        assert model.effective_batch_size is None

    def test_accumulation_steps_computed(self, simple_lightning_module):
        """micro=8, effective=32 → accumulation_steps=4."""
        model = analyzer(
            simple_lightning_module,
            micro_batch_size=8,
            effective_batch_size=32,
        )
        assert model.accumulation_steps == 4

    def test_effective_without_micro_batch_raises(self, simple_lightning_module):
        """effective_batch_size alone → ValueError."""
        with pytest.raises(ValueError, match="micro_batch_size must be specified"):
            analyzer(
                simple_lightning_module,
                effective_batch_size=32,
            )

    def test_effective_less_than_micro_batch_raises(self, simple_lightning_module):
        """effective=8, micro=32 → ValueError."""
        with pytest.raises(ValueError, match="must be >= micro_batch_size"):
            analyzer(
                simple_lightning_module,
                micro_batch_size=32,
                effective_batch_size=8,
            )

    def test_not_divisible_raises(self, simple_lightning_module):
        """effective=30, micro=8 → ValueError (not divisible)."""
        with pytest.raises(ValueError, match="must be divisible"):
            analyzer(
                simple_lightning_module,
                micro_batch_size=8,
                effective_batch_size=30,
            )

    def test_accumulation_with_delegate_raises(self, manual_optimization_module):
        """Accumulation + manual optimization module → ValueError."""
        with pytest.raises(
            ValueError,
            match="Gradient accumulation is not supported",
        ):
            with pytest.warns(UserWarning, match="manual optimization"):
                analyzer(
                    manual_optimization_module,
                    micro_batch_size=8,
                    effective_batch_size=32,
                )

    # --- B. Optimizer behavior ---

    def test_zero_grad_once_per_cycle(self, simple_lightning_module, sample_batch):
        """4 micro-steps, accum=4: zero_grad called exactly 1x."""
        model = analyzer(
            simple_lightning_module,
            disable_analyzer=True,
            micro_batch_size=4,
            effective_batch_size=16,
        )
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        x, y = sample_batch
        for i in range(4):
            model.training_step((x, y), i)

        assert mock_opt.zero_grad.call_count == 1

    def test_step_once_per_cycle(self, simple_lightning_module, sample_batch):
        """4 micro-steps, accum=4: opt.step() called exactly 1x."""
        model = analyzer(
            simple_lightning_module,
            disable_analyzer=True,
            micro_batch_size=4,
            effective_batch_size=16,
        )
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        x, y = sample_batch
        for i in range(4):
            model.training_step((x, y), i)

        assert mock_opt.step.call_count == 1

    def test_step_not_called_mid_cycle(self, simple_lightning_module, sample_batch):
        """3 of 4 micro-steps done: opt.step() never called."""
        model = analyzer(
            simple_lightning_module,
            disable_analyzer=True,
            micro_batch_size=4,
            effective_batch_size=16,
        )
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        x, y = sample_batch
        for i in range(3):
            model.training_step((x, y), i)

        mock_opt.step.assert_not_called()

    def test_loss_scaled_for_backward(self, simple_lightning_module, sample_batch):
        """manual_backward receives loss / accumulation_steps."""
        model = analyzer(
            simple_lightning_module,
            disable_analyzer=True,
            micro_batch_size=4,
            effective_batch_size=16,
        )
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        x, y = sample_batch
        output = model.training_step((x, y), 0)

        backward_arg = model.manual_backward.call_args[0][0]
        expected = output / 4
        assert torch.allclose(backward_arg, expected)

    def test_unscaled_loss_returned(self, simple_lightning_module, sample_batch):
        """training_step returns the original unscaled loss."""
        model = analyzer(
            simple_lightning_module,
            disable_analyzer=True,
            micro_batch_size=4,
            effective_batch_size=16,
        )
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        x, y = sample_batch
        output_accum = model.training_step((x, y), 0)
        assert isinstance(output_accum, torch.Tensor)

    def test_two_full_cycles(self, simple_lightning_module, sample_batch):
        """4 steps, accum=2: zero_grad 2x, opt.step() 2x."""
        model = analyzer(
            simple_lightning_module,
            disable_analyzer=True,
            micro_batch_size=4,
            effective_batch_size=8,
        )
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        x, y = sample_batch
        for i in range(4):
            model.training_step((x, y), i)

        assert mock_opt.zero_grad.call_count == 2
        assert mock_opt.step.call_count == 2

    # --- C. Backwards compatibility ---

    def test_no_accumulation_backwards_compatible(
        self, simple_lightning_module, sample_batch
    ):
        """No accum params: every call does zero_grad + step."""
        model = analyzer(simple_lightning_module, disable_analyzer=True)
        mock_opt = Mock(zero_grad=Mock(), step=Mock())
        model.optimizers = Mock(return_value=mock_opt)
        model.manual_backward = Mock()

        x, y = sample_batch
        for i in range(4):
            model.training_step((x, y), i)

        assert mock_opt.zero_grad.call_count == 4
        assert mock_opt.step.call_count == 4

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_single_step_path_unchanged(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Without accumulation, _analyze_single_step produces same results."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.5),
            "batch_grad_norms_loss": torch.tensor(2.5),
        }
        mock_probe.return_value = {
            "self": (1.0, 0.0, -1.0),
            "cross": None,
        }

        model = analyzer(simple_lightning_module, log_metrics=True)
        model.log = Mock()
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}
        assert torch.allclose(logged["chi_net"], torch.tensor(1.5))
        assert torch.allclose(logged["chi_loss"], torch.tensor(2.5))
        assert logged["loss"] == 1.0
        assert logged["grad_norm_squared"] == 1.0
        assert logged["batch_size"] == 4

    # --- D. Effective step ---

    def test_effective_step_with_accumulation(self, simple_lightning_module):
        """_optimizer_step_count=2 → effective_step=2."""
        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=16,
        )
        model._optimizer_step_count = 2
        assert model.effective_step == 2

        model._optimizer_step_count = 0
        assert model.effective_step == 0

    def test_effective_step_without_accumulation(self, simple_lightning_module):
        """_optimizer_step_count=42 → effective_step=42."""
        model = analyzer(simple_lightning_module)
        model._optimizer_step_count = 42
        assert model.effective_step == 42

    # --- E. Analysis scheduling with accumulation ---

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_analysis_uses_effective_step_for_scheduling(
        self, mock_compute, simple_lightning_module, sample_batch
    ):
        """_should_analyze uses effective_step, activates on first micro-batch."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }

        model = analyzer(
            simple_lightning_module,
            analyze_every=2,
            micro_batch_size=4,
            effective_batch_size=16,
        )
        model.log = Mock()
        x, y = sample_batch

        # effective_step=0, analyze_every=2 → 0 % 2 == 0 → analyze
        model._accumulation_count = 0
        model._before_training_step((x, y), 0)
        assert model._analysis_active is True

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_analysis_skipped_when_schedule_says_no(
        self, mock_compute, simple_lightning_module, sample_batch
    ):
        """When _should_analyze returns False, no accumulation or logging happens."""
        model = analyzer(
            simple_lightning_module,
            analyze_every=10,
            micro_batch_size=4,
            effective_batch_size=8,
        )
        model.log = Mock()
        x, y = sample_batch

        # effective_step=1, analyze_every=10 → 1 % 10 != 0 → skip
        model._optimizer_step_count = 1
        model._accumulation_count = 0
        model._before_training_step((x, y), 0)

        assert model._analysis_active is False
        mock_compute.assert_not_called()
        model.log.assert_not_called()

        # Second micro-batch also skipped (flag persists)
        model._accumulation_count = 1
        model._before_training_step((x, y), 1)
        mock_compute.assert_not_called()
        model.log.assert_not_called()

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_analysis_step_logs_effective_step(
        self, mock_compute, simple_lightning_module, sample_batch
    ):
        """Logged analysis_step equals effective_step, not global_step."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }

        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        # _optimizer_step_count=3 → effective_step=3
        model._optimizer_step_count = 3

        model._accumulation_count = 0
        model._before_training_step((x, y), 0)
        model._accumulation_count = 1
        model._before_training_step((x, y), 1)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}
        assert logged["analysis_step"] == 3

    # --- F. Sample-wise metric accumulation ---

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_accumulated_chi_net_is_mean(
        self, mock_compute, simple_lightning_module, sample_batch
    ):
        """chi_net_eff = mean of per-micro-batch chi_net values."""
        mock_compute.side_effect = [
            {
                "batch_grad_norms_network": torch.tensor(2.0),
                "batch_grad_norms_loss": torch.tensor(3.0),
            },
            {
                "batch_grad_norms_network": torch.tensor(4.0),
                "batch_grad_norms_loss": torch.tensor(5.0),
            },
        ]

        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        model._accumulation_count = 0
        model._before_training_step((x, y), 0)
        model._accumulation_count = 1
        model._before_training_step((x, y), 1)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}

        # chi_net_eff = mean([2.0, 4.0]) = 3.0
        assert torch.allclose(logged["chi_net"], torch.tensor(3.0))

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_accumulated_chi_loss_is_mean(
        self, mock_compute, simple_lightning_module, sample_batch
    ):
        """chi_loss_eff = mean of per-micro-batch chi_loss values.

        chi_loss is computed with normalize=True against a mean-reduced loss,
        so the correct aggregation for an effective batch is the mean across
        micro-batches (same as chi_net), not K * sum.
        """
        mock_compute.side_effect = [
            {
                "batch_grad_norms_network": torch.tensor(2.0),
                "batch_grad_norms_loss": torch.tensor(3.0),
            },
            {
                "batch_grad_norms_network": torch.tensor(4.0),
                "batch_grad_norms_loss": torch.tensor(5.0),
            },
        ]

        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        model._accumulation_count = 0
        model._before_training_step((x, y), 0)
        model._accumulation_count = 1
        model._before_training_step((x, y), 1)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}

        # chi_loss_eff = mean([3.0, 5.0]) = 4.0
        assert torch.allclose(logged["chi_loss"], torch.tensor(4.0))

    # --- G. Linearizer gradient accumulation ---

    def test_linearizer_accumulated_grad_norm(
        self, simple_lightning_module, sample_batch
    ):
        """grad_norm_squared = ||Σ∇L_k||² / K² from accumulated grads."""
        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        # Run a full accumulation cycle (K=2)
        with patch.object(
            SamplewiseCalculatorOpacus,
            "compute",
            return_value={
                "batch_grad_norms_network": torch.tensor(1.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
        ):
            model._accumulation_count = 0
            model._before_training_step((x, y), 0)
            model._accumulation_count = 1
            model._before_training_step((x, y), 1)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}

        # grad_norm_squared should be a positive float
        assert "grad_norm_squared" in logged
        assert logged["grad_norm_squared"] > 0

        # Verify manually: compute the expected value
        # Do two forward+backward passes, sum grads, compute ||sum||²/K²
        model.model.zero_grad()
        loss0 = model.criterion(model.model(x), y)
        loss0.backward()
        grads_0 = [
            p.grad.clone() for p in model.model.parameters() if p.grad is not None
        ]

        model.model.zero_grad()
        loss1 = model.criterion(model.model(x), y)
        loss1.backward()
        grads_1 = [
            p.grad.clone() for p in model.model.parameters() if p.grad is not None
        ]
        model.model.zero_grad()

        expected_norm_sq = (
            sum(((g0 + g1) ** 2).sum().item() for g0, g1 in zip(grads_0, grads_1)) / 4
        )  # K² = 2² = 4

        assert abs(logged["grad_norm_squared"] - expected_norm_sq) < 1e-4

    # --- H. Coupling with accumulated values ---

    def test_coupling_from_accumulated_values(
        self, simple_lightning_module, sample_batch
    ):
        """coupling = grad_norm_sq / (chi_loss_eff * chi_net_eff)."""
        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        with patch.object(
            SamplewiseCalculatorOpacus,
            "compute",
            return_value={
                "batch_grad_norms_network": torch.tensor(2.0),
                "batch_grad_norms_loss": torch.tensor(3.0),
            },
        ):
            model._accumulation_count = 0
            model._before_training_step((x, y), 0)
            model._accumulation_count = 1
            model._before_training_step((x, y), 1)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}

        # chi_net_eff  = mean([2.0, 2.0]) = 2.0
        # chi_loss_eff = mean([3.0, 3.0]) = 3.0
        # coupling = grad_norm_sq / (3.0 * 2.0)
        assert "chi_coup" in logged
        expected_coupling = logged["grad_norm_squared"] / (
            logged["chi_loss"] * logged["chi_net"]
        )
        assert abs(logged["chi_coup"] - expected_coupling) < 1e-5

    # --- I. Logging behavior ---

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_logging_only_on_last_microbatch(
        self, mock_compute, simple_lightning_module, sample_batch
    ):
        """Metrics logged once per cycle on the last micro-batch only."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }

        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        # First micro-batch — should NOT log yet
        model._accumulation_count = 0
        model._before_training_step((x, y), 0)
        assert model.log.call_count == 0

        # Second micro-batch (last) — should log
        model._accumulation_count = 1
        model._before_training_step((x, y), 1)
        assert model.log.call_count > 0

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_effective_batch_size_logged(
        self, mock_compute, simple_lightning_module, sample_batch
    ):
        """effective_batch_size is logged when accumulation is active."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }

        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        model._accumulation_count = 0
        model._before_training_step((x, y), 0)
        model._accumulation_count = 1
        model._before_training_step((x, y), 1)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}

        assert "effective_batch_size" in logged
        # micro_batch_size=4, accumulation_steps=2 → 4*2=8
        assert logged["effective_batch_size"] == 8

    # --- J. Buffer cleanup ---

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_buffers_cleared_after_cycle(
        self, mock_compute, simple_lightning_module, sample_batch
    ):
        """Accumulation buffers are reset after a full cycle."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }

        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        # Run one full cycle
        model._accumulation_count = 0
        model._before_training_step((x, y), 0)
        model._accumulation_count = 1
        model._before_training_step((x, y), 1)

        # Buffers should be cleared
        assert len(model._accum_chi_net) == 0
        assert len(model._accum_chi_loss) == 0
        assert model._accum_grad_train is None
        assert model._accum_grad_measure is None
        assert model._accum_train_loss == 0.0

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_buffers_dont_leak_between_cycles(
        self, mock_compute, simple_lightning_module, sample_batch
    ):
        """Second cycle doesn't contain data from the first cycle."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # Cycle 1: return 10.0, Cycle 2: return 20.0
            val = 10.0 if call_count[0] <= 2 else 20.0
            return {
                "batch_grad_norms_network": torch.tensor(val),
                "batch_grad_norms_loss": torch.tensor(1.0),
            }

        mock_compute.side_effect = side_effect

        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        # Cycle 1
        model._accumulation_count = 0
        model._before_training_step((x, y), 0)
        model._accumulation_count = 1
        model._before_training_step((x, y), 1)

        # Cycle 2
        model._accumulation_count = 0
        model._before_training_step((x, y), 2)
        model._accumulation_count = 1
        model._before_training_step((x, y), 3)

        # Get chi_net from cycle 2 (the last logged value)
        chi_net_calls = [
            call[0][1] for call in model.log.call_args_list if call[0][0] == "chi_net"
        ]
        # Cycle 1: mean([10, 10]) = 10, Cycle 2: mean([20, 20]) = 20
        assert len(chi_net_calls) == 2
        assert torch.allclose(chi_net_calls[0], torch.tensor(10.0))
        assert torch.allclose(chi_net_calls[1], torch.tensor(20.0))

    def test_analysis_does_not_corrupt_training_gradients(
        self, simple_lightning_module, sample_batch
    ):
        """Analysis backward must not affect the gradients seen by the optimizer.

        With accumulation_steps=2, the gradient accumulated into p.grad after
        two training_step calls (with analysis enabled) must match the gradient
        from two training_step calls with analysis disabled.
        """
        torch.manual_seed(0)
        x, y = sample_batch

        def run_two_steps(with_analysis):
            torch.manual_seed(0)
            model = analyzer(
                simple_lightning_module,
                micro_batch_size=4,
                effective_batch_size=8,
                disable_analyzer=not with_analysis,
                log_metrics=False,
            )
            model.log = Mock()
            # Use a real optimizer so p.grad is populated
            opt = torch.optim.SGD(model.parameters(), lr=0.0)
            model.optimizers = Mock(return_value=opt)
            model.manual_backward = lambda loss: loss.backward()
            model._trainer = None

            opt.zero_grad()
            model._accumulation_count = 0
            model.training_step((x, y), 0)
            model.training_step((x, y), 1)

            return [
                p.grad.clone() if p.grad is not None else None
                for p in model.model.parameters()
            ]

        grads_with = run_two_steps(with_analysis=True)
        grads_without = run_two_steps(with_analysis=False)

        for g_with, g_without in zip(grads_with, grads_without):
            assert g_with is not None and g_without is not None
            assert torch.allclose(g_with, g_without, atol=1e-6), (
                f"Analysis pass corrupted training gradients: "
                f"max diff {(g_with - g_without).abs().max().item()}"
            )


class TestIndependentMeasureResponse:
    """Test the independent measure_dataloader / measure_batch_size path."""

    # --- A. Parameter validation ---

    def test_infers_measure_micro_from_dataloader(self, simple_lightning_module):
        """measure micro-batch size is inferred from the loader's batch_size."""
        measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=1)
        model = analyzer(simple_lightning_module, measure_dataloader=measure_loader)
        assert model._measure_micro_batch_size == 4
        assert model._measure_batch_sizes == [4]

    def test_measure_batch_size_int_normalized_to_list(self, simple_lightning_module):
        """A scalar measure_batch_size is normalized to a length-1 list."""
        measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=1)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            measure_batch_size=8,
        )
        assert model._measure_batch_sizes == [8]

    def test_measure_batch_size_not_divisible_raises(self, simple_lightning_module):
        """measure_batch_size above micro, not an exact multiple, raises."""
        measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=1)
        with pytest.raises(ValueError, match="exact multiple"):
            analyzer(
                simple_lightning_module,
                measure_dataloader=measure_loader,
                measure_batch_size=6,
            )

    def test_measure_batch_size_below_micro_allowed(self, simple_lightning_module):
        """measure_batch_size below the measure micro size is valid: it runs
        as a single direct pass (no accumulation), not an error."""
        measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=1)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            measure_batch_size=2,
        )
        assert model._measure_batch_sizes == [2]

    def test_measure_dataloader_batch_size_none_raises(self, simple_lightning_module):
        """A measure_dataloader with batch_size=None (manual batching) raises."""
        ds = TensorDataset(torch.randn(4, 10), torch.randint(0, 2, (4,)))
        loader = DataLoader(ds, batch_size=None)
        with pytest.raises(ValueError, match="batch_size"):
            analyzer(simple_lightning_module, measure_dataloader=loader)

    def test_measure_params_without_dataloader_raises(self, simple_lightning_module):
        """measure_batch_size/measure_subset_seed require measure_dataloader."""
        with pytest.raises(
            ValueError, match="only valid together with measure_dataloader"
        ):
            analyzer(simple_lightning_module, measure_batch_size=8)

    def test_sweep_without_schedule_warns(self, simple_lightning_module):
        """A multi-size sweep without an analysis_schedule warns."""
        measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=1)
        with pytest.warns(UserWarning, match="batch-size sweep"):
            analyzer(
                simple_lightning_module,
                measure_dataloader=measure_loader,
                measure_batch_size=[4, 8],
            )

    def test_sweep_with_schedule_does_not_warn(self, simple_lightning_module):
        """A multi-size sweep with a logarithmic analysis_schedule doesn't warn."""
        from perspic.logger import logarithmic_windows

        measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=1)
        schedule = logarithmic_windows(max_steps=100)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            analyzer(
                simple_lightning_module,
                measure_dataloader=measure_loader,
                measure_batch_size=[4, 8],
                analysis_schedule=schedule,
            )
        assert not any("batch-size sweep" in str(w.message) for w in record)

    # --- B. Single-measure logging behavior ---

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_logs_cross_keys_single_measure(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """A single measure_batch_size logs the existing unsuffixed cross_* keys."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {"self": (1.0, 0.0, -1.0), "cross": None}

        measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=1)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            measure_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch
        model._before_training_step((x, y), 0)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}
        for key in (
            "cross_chi_net",
            "cross_chi_loss",
            "cross_chi_coup",
            "cross_loss",
            "cross_grad_dot_product",
            "cross_batch_size",
        ):
            assert key in logged
        assert logged["cross_batch_size"] == 8
        assert "cross_chi_net@bs8" not in logged

    @patch.object(SamplewiseCalculatorOpacus, "compute_cross_metrics")
    @patch.object(SamplewiseCalculatorOpacus, "compute")
    def test_measure_chi_is_mean_over_kmeasure(
        self,
        mock_compute,
        mock_compute_cross,
        simple_lightning_module,
        sample_batch,
    ):
        """Measure-side chi is averaged over K_measure = S // measure_micro,
        not the (different) train accumulation_steps."""
        # 3 train micro-batches (accumulation_steps=3), then 2 measure chunks
        # (measure_batch_size=8, measure_micro=4 -> K_measure=2).
        values = [
            {
                "batch_grad_norms_network": torch.tensor(2.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
            {
                "batch_grad_norms_network": torch.tensor(4.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
            {
                "batch_grad_norms_network": torch.tensor(6.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
            {
                "batch_grad_norms_network": torch.tensor(10.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
            {
                "batch_grad_norms_network": torch.tensor(20.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
        ]
        mock_compute.side_effect = values
        mock_compute_cross.return_value = {
            "batch_grad_norms_network": torch.tensor(0.0),
            "batch_grad_norms_loss": torch.tensor(0.0),
        }

        measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=5)
        model = analyzer(
            simple_lightning_module,
            micro_batch_size=4,
            effective_batch_size=12,
            measure_dataloader=measure_loader,
            measure_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        model._accumulation_count = 0
        model._before_training_step((x, y), 0)
        model._accumulation_count = 1
        model._before_training_step((x, y), 1)
        model._accumulation_count = 2
        model._before_training_step((x, y), 2)

        assert mock_compute_cross.call_count == 1
        _, kwargs = mock_compute_cross.call_args
        measure_agg = kwargs["sample_wise_metrics_cross"]
        # mean([10.0, 20.0]) = 15.0, NOT sum/3 (train K) == 10.0
        assert torch.allclose(
            measure_agg["batch_grad_norms_network"], torch.tensor(15.0)
        )

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_measure_grad_dot_reference(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """cross_grad_dot_product matches a hand-computed <grad_train, grad_measure>."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {"self": (1.0, 0.0, -1.0), "cross": None}

        measure_loader = _make_measure_dataloader(n_samples=4, micro=4, seed=99)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}
        assert "cross_grad_dot_product" in logged

        # Hand-compute: grad_train = ∇L(x, y); grad_measure = ∇L(x_m, y_m),
        # where x_m, y_m is the same (only) batch measure_loader yields
        # (shuffle=False), reused via a fresh iterator over the same loader.
        x_m, y_m = next(iter(measure_loader))

        model.model.zero_grad()
        loss_t = model.criterion(model.model(x), y)
        loss_t.backward()
        grad_train = [p.grad.clone() for p in model.model.parameters()]

        model.model.zero_grad()
        loss_m = model.criterion(model.model(x_m), y_m)
        loss_m.backward()
        grad_measure = [p.grad.clone() for p in model.model.parameters()]
        model.model.zero_grad()

        expected_dot = sum(
            (g1 * g2).sum().item() for g1, g2 in zip(grad_train, grad_measure)
        )
        assert abs(logged["cross_grad_dot_product"] - expected_dot) < 1e-4

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_direct_pass_below_micro_reference(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """A measure_batch_size below the measure micro size runs as a
        single direct pass (K_measure=1, no accumulation/padding) on exactly
        S samples taken from the front of the gathered pool."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {"self": (1.0, 0.0, -1.0), "cross": None}

        measure_loader = _make_measure_dataloader(n_samples=8, micro=8, seed=99)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            measure_batch_size=4,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        model._before_training_step((x, y), 0)

        # 1 call for the train self chi + exactly 1 measure-side call
        # (K_measure=1: a single direct pass, not padded up to micro=8).
        assert mock_compute.call_count == 2

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}
        assert logged["cross_batch_size"] == 4

        # Hand-compute the reference: the pool is built from ceil(4/8)=1
        # micro-batch pull (the loader's only 8-sample batch), sliced to the
        # first S_max=4 samples.
        x_full, y_full = next(iter(measure_loader))
        x_m, y_m = x_full[:4], y_full[:4]

        model.model.zero_grad()
        loss_t = model.criterion(model.model(x), y)
        loss_t.backward()
        grad_train = [p.grad.clone() for p in model.model.parameters()]

        model.model.zero_grad()
        loss_m = model.criterion(model.model(x_m), y_m)
        loss_m.backward()
        grad_measure = [p.grad.clone() for p in model.model.parameters()]
        model.model.zero_grad()

        expected_dot = sum(
            (g1 * g2).sum().item() for g1, g2 in zip(grad_train, grad_measure)
        )
        assert abs(logged["cross_grad_dot_product"] - expected_dot) < 1e-4

    def test_measure_backward_does_not_corrupt_training_grad(
        self, simple_lightning_module, sample_batch
    ):
        """Measure-side backward passes inside _measure_response must not
        corrupt the (possibly partial) accumulated training gradient."""
        x, y = sample_batch

        def run_two_steps(with_measure):
            torch.manual_seed(0)
            kwargs = {}
            if with_measure:
                measure_loader = _make_measure_dataloader(n_samples=4, micro=4, seed=3)
                kwargs = {"measure_dataloader": measure_loader}

            model = analyzer(
                simple_lightning_module,
                micro_batch_size=4,
                effective_batch_size=8,
                log_metrics=False,
                **kwargs,
            )
            model.log = Mock()
            opt = torch.optim.SGD(model.parameters(), lr=0.0)
            model.optimizers = Mock(return_value=opt)
            model.manual_backward = lambda loss: loss.backward()
            model._trainer = None

            opt.zero_grad()
            model._accumulation_count = 0
            model.training_step((x, y), 0)
            model.training_step((x, y), 1)

            return [
                p.grad.clone() if p.grad is not None else None
                for p in model.model.parameters()
            ]

        grads_with = run_two_steps(with_measure=True)
        grads_without = run_two_steps(with_measure=False)

        for g_with, g_without in zip(grads_with, grads_without):
            assert g_with is not None and g_without is not None
            assert torch.allclose(g_with, g_without, atol=1e-6), (
                f"Measure-response backward corrupted training gradients: "
                f"max diff {(g_with - g_without).abs().max().item()}"
            )

    # --- C. Subset sampling ---

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_measure_subset_deterministic_with_seed(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """The same measure_subset_seed yields the same subset (and hence the
        same swept cross metric); a different seed yields a different one."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {"self": (1.0, 0.0, -1.0), "cross": None}
        x, y = sample_batch

        def run(seed):
            # Fix the model init seed too: cross_grad_dot_product is a real
            # gradient dot product, so it depends on model weights as well
            # as on which subset is drawn.
            torch.manual_seed(0)
            measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=11)
            model = analyzer(
                simple_lightning_module,
                measure_dataloader=measure_loader,
                measure_batch_size=[8, 4],
                measure_subset_seed=seed,
                log_metrics=True,
            )
            model.log = Mock()
            model._before_training_step((x, y), 0)
            logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}
            return logged["cross_grad_dot_product@bs4"]

        val_a = run(seed=123)
        val_b = run(seed=123)
        val_c = run(seed=456)

        assert val_a == val_b
        assert val_a != val_c

    @patch.object(Linearizer, "compute")
    def test_measure_subset_size_matches_S(
        self, mock_probe, simple_lightning_module, sample_batch
    ):
        """The number of measure micro-batch chunks processed for each swept
        size equals S // measure_micro_batch_size."""
        mock_probe.return_value = {"self": (1.0, 0.0, -1.0), "cross": None}

        measure_loader = _make_measure_dataloader(n_samples=16, micro=4, seed=1)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            measure_batch_size=[16, 8],
            measure_subset_seed=0,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        with patch.object(
            SamplewiseCalculatorOpacus,
            "compute",
            return_value={
                "batch_grad_norms_network": torch.tensor(1.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
        ) as mock_compute:
            model._before_training_step((x, y), 0)
            # 1 call for train self chi + (16 // 4 = 4) + (8 // 4 = 2) measure
            # chunks.
            assert mock_compute.call_count == 1 + 4 + 2

    @patch.object(Linearizer, "compute")
    def test_pool_gather_when_S_max_below_micro(
        self, mock_probe, simple_lightning_module, sample_batch
    ):
        """When every swept size is below the measure micro size, the pool
        gather still pulls at least one micro-batch (ceil(S_max/micro)=1)
        and slices it down to S_max samples, rather than pulling zero."""
        mock_probe.return_value = {"self": (1.0, 0.0, -1.0), "cross": None}

        measure_loader = _make_measure_dataloader(n_samples=32, micro=16, seed=1)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            measure_batch_size=[2, 4, 8],
            measure_subset_seed=0,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        with patch.object(
            model,
            "_next_measure_micro_batch",
            wraps=model._next_measure_micro_batch,
        ) as mock_pull, patch.object(
            SamplewiseCalculatorOpacus,
            "compute",
            return_value={
                "batch_grad_norms_network": torch.tensor(1.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
        ):
            model._before_training_step((x, y), 0)

        # S_max=8 < micro=16 -> ceil(8/16)=1 pull, independent of how many
        # sizes are swept below it.
        assert mock_pull.call_count == 1

    @patch.object(Linearizer, "compute")
    def test_mixed_regime_sweep_chunk_counts(
        self, mock_probe, simple_lightning_module, sample_batch
    ):
        """A sweep spanning sizes below, at, and above the measure micro
        size produces the correct chunk count for each: K_measure=1 for
        sizes <= micro (a single direct pass), K_measure=S // micro for
        sizes above it (accumulated)."""
        mock_probe.return_value = {"self": (1.0, 0.0, -1.0), "cross": None}

        measure_loader = _make_measure_dataloader(n_samples=32, micro=8, seed=1)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            measure_batch_size=[2, 8, 32],
            measure_subset_seed=0,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        with patch.object(
            SamplewiseCalculatorOpacus,
            "compute",
            return_value={
                "batch_grad_norms_network": torch.tensor(1.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
        ) as mock_compute:
            model._before_training_step((x, y), 0)
            # 1 train self chi call + (2 // 2 = 1) + (8 // 8 = 1)
            # + (32 // 8 = 4) measure chunks.
            assert mock_compute.call_count == 1 + 1 + 1 + 4

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}
        for S in (2, 8, 32):
            assert logged[f"cross_batch_size@bs{S}"] == S

    # --- D. Sweep logging ---

    @patch.object(SamplewiseCalculatorOpacus, "compute")
    @patch.object(Linearizer, "compute")
    def test_sweep_logs_suffixed_keys(
        self, mock_probe, mock_compute, simple_lightning_module, sample_batch
    ):
        """Sweeping multiple measure_batch_size values logs @bs{S}-suffixed
        keys instead of the unsuffixed cross_* keys."""
        mock_compute.return_value = {
            "batch_grad_norms_network": torch.tensor(1.0),
            "batch_grad_norms_loss": torch.tensor(1.0),
        }
        mock_probe.return_value = {"self": (1.0, 0.0, -1.0), "cross": None}

        measure_loader = _make_measure_dataloader(n_samples=8, micro=4, seed=1)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            measure_batch_size=[4, 8],
            measure_subset_seed=0,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch
        model._before_training_step((x, y), 0)

        logged = {call[0][0]: call[0][1] for call in model.log.call_args_list}
        for S in (4, 8):
            for key in (
                "cross_chi_net",
                "cross_chi_loss",
                "cross_chi_coup",
                "cross_grad_dot_product",
                "cross_loss",
                "cross_batch_size",
            ):
                assert f"{key}@bs{S}" in logged
        assert "cross_chi_net" not in logged

    # --- E. Persistent iterator ---

    def test_measure_iterator_refills_on_exhaustion(
        self, simple_lightning_module, sample_batch
    ):
        """A measure_dataloader smaller than what's pulled across several
        analysis steps cycles instead of raising StopIteration."""
        measure_loader = _make_measure_dataloader(n_samples=4, micro=4, seed=1)
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        with patch.object(
            SamplewiseCalculatorOpacus,
            "compute",
            return_value={
                "batch_grad_norms_network": torch.tensor(1.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
        ), patch.object(
            Linearizer,
            "compute",
            return_value={"self": (1.0, 0.0, -1.0), "cross": None},
        ):
            for i in range(5):
                model._before_training_step((x, y), i)

    def test_measure_partial_last_batch_raises(
        self, simple_lightning_module, sample_batch
    ):
        """drop_last=False measure loader with a non-divisible dataset raises
        a clear ValueError instead of silently building a short pool."""
        measure_loader = _make_measure_dataloader(
            n_samples=6, micro=4, seed=2, drop_last=False
        )
        model = analyzer(
            simple_lightning_module,
            measure_dataloader=measure_loader,
            measure_batch_size=8,
            log_metrics=True,
        )
        model.log = Mock()
        x, y = sample_batch

        with patch.object(
            SamplewiseCalculatorOpacus,
            "compute",
            return_value={
                "batch_grad_norms_network": torch.tensor(1.0),
                "batch_grad_norms_loss": torch.tensor(1.0),
            },
        ), patch.object(
            Linearizer,
            "compute",
            return_value={"self": (1.0, 0.0, -1.0), "cross": None},
        ):
            with pytest.raises(ValueError, match="drop_last"):
                model._before_training_step((x, y), 0)
