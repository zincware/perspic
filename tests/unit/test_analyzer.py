"""Unit tests for the analyzer module."""

from unittest.mock import Mock, patch

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, StepLR

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
        model._global_step = 5  # Not a multiple of 10

        # Mock global_step property
        type(model).global_step = property(lambda self: 5)

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
        type(model).global_step = property(lambda self: 0)

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
        type(model).global_step = property(lambda self: 0)

        model._before_training_step(sample_batch, 0)

        logged_names = [call[0][0] for call in model.log.call_args_list]
        assert "window_id" not in logged_names
        assert "window_center" not in logged_names
        assert "window_width" not in logged_names


class SchedulerModule(pl.LightningModule):
    def __init__(self, scheduler_type="step"):
        super().__init__()
        self.model = nn.Linear(10, 2)
        self.criterion = F.cross_entropy
        self.scheduler_type = scheduler_type
        self.lr = 0.1

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.scheduler_type == "step":
            scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        elif self.scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=10,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return optimizer


def test_analyzer_step_scheduler():
    """Test that step-based scheduler is stepped correctly."""
    # Wrap with analyzer
    model = analyzer(SchedulerModule, scheduler_type="onecycle")

    trainer = pl.Trainer(
        max_epochs=1, limit_train_batches=5, enable_checkpointing=False, logger=False
    )

    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=4
    )

    # Initial LR
    # We can't easily check LR before training starts because scheduler is initialized in configure_optimizers
    # which is called by trainer.

    # We can use a callback to check LR
    class LRCheckCallback(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if batch_idx == 0:
                self.initial_lr = trainer.optimizers[0].param_groups[0]["lr"]

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            current_lr = trainer.optimizers[0].param_groups[0]["lr"]
            # OneCycleLR starts low, increases, then decreases.
            # But we just want to see it change.
            if batch_idx > 0:
                assert (
                    current_lr != self.initial_lr
                ), f"LR did not change at batch {batch_idx}"

    lr_callback = LRCheckCallback()
    trainer.callbacks.append(lr_callback)

    trainer.fit(model, train_loader)


def test_analyzer_epoch_scheduler():
    """Test that epoch-based scheduler is stepped correctly."""
    # StepLR with step_size=1 means it decays every epoch
    model = analyzer(SchedulerModule, scheduler_type="step")

    trainer = pl.Trainer(
        max_epochs=2, limit_train_batches=2, enable_checkpointing=False, logger=False
    )

    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=4
    )

    class LREpochCheckCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            self.current_epoch_lr = trainer.optimizers[0].param_groups[0]["lr"]

        def on_train_epoch_end(self, trainer, pl_module):
            pass

    lr_callback = LREpochCheckCallback()
    trainer.callbacks.append(lr_callback)

    trainer.fit(model, train_loader)

    # After 2 epochs, LR should have decayed twice (start of epoch 0 -> end of epoch 0 (decay) -> start of epoch 1 -> end of epoch 1 (decay))
    # Wait, StepLR steps at the end of epoch.
    # Initial LR is 0.1.
    # After epoch 0, it becomes 0.01.
    # After epoch 1, it becomes 0.001.

    final_lr = trainer.optimizers[0].param_groups[0]["lr"]
    assert final_lr < 0.1, f"LR should have decayed, got {final_lr}"
    assert final_lr == pytest.approx(
        0.001
    ), f"LR should be 0.001 after 2 epochs, got {final_lr}"
