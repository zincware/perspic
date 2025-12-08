import warnings
from typing import Optional

import pytorch_lightning as pl
import torch

from perspic.calculator.coupling import CouplingCalculator
from perspic.calculator.linearizer import Linearizer
from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus
from perspic.logger import LogarithmicWindowSchedule
from perspic.utils import BatchStatSnapshot


def analyzer(
    lightning_module: pl.LightningModule,
    sample_wise_engine: Optional[str] = "opacus",
    disable_analyzer: bool = False,
    log_metrics: bool = True,
    opacus_strict: bool = False,
    analyze_every: Optional[int] = None,
    analysis_schedule: Optional[LogarithmicWindowSchedule] = None,
    cross_response_loader: Optional[torch.utils.data.DataLoader] = None,
    **model_kwargs,
):
    """Factory function that wraps a LightningModule with analysis capabilities.

    This function creates an Analyzer class that inherits from the provided
    LightningModule and adds functionality for computing sample-wise gradients
    and probing linearization properties during training.

    Args:
        lightning_module: A PyTorch Lightning module class to wrap with
            analysis features.
        sample_wise_engine: Engine for computing sample-wise gradients.
            Options: 'functorch' or 'opacus'. Defaults to 'opacus'.
        disable_analyzer: If True, wraps the module without adding analysis
            capabilities. Defaults to False. Mainly for testing purposes.
        log_metrics: If True, logs analysis metrics during training. Defaults to True.
        opacus_strict: If True and using 'opacus' engine, Opacus will validate
            that all layers are supported for per-sample gradient computation.
            Defaults to False.
        analyze_every: If provided, run analysis every N steps (0, N, 2N, ...).
            If None and no analysis_schedule, runs every step.
        analysis_schedule: A LogarithmicWindowSchedule that defines which steps
            to analyze. Created via `logarithmic_windows()`. If provided,
            analysis runs only at the scheduled steps.
            If both analyze_every and analysis_schedule are provided, analysis_schedule
            takes precedence.
        cross_response_loader: Optional DataLoader for computing cross-batch
            linear response. When provided, the linearizer computes both the
            "self" response (gradient on training batch) and "cross" response
            (gradient dot product between training batch and cross batch).
        **model_kwargs: Additional keyword arguments passed to the
            LightningModule constructor.

    Returns:
        An initialized Analyzer instance that wraps the provided
        LightningModule.

    Raises:
        ValueError: If sample_wise_engine is not 'functorch' or 'opacus'.
        AttributeError: If the wrapped module doesn't have a 'criterion'
            attribute.

    Examples:
        # Analyze every step
        model = analyzer(MyModule, model=backbone, lr=0.01)

        # Analyze every 100 steps
        model = analyzer(MyModule, analyze_every=100, model=backbone, lr=0.01)

        # Analyze at logarithmically spaced windows
        from perspic import logarithmic_windows
        schedule = logarithmic_windows(max_steps=10000, points_per_decade=5)
        model = analyzer(MyModule, analysis_schedule=schedule, model=backbone, lr=0.01)

        # Analyze with cross-batch response
        cross_loader = DataLoader(cross_dataset, batch_size=32)
        model = analyzer(MyModule, cross_response_loader=cross_loader, model=backbone, lr=0.01)

    Note:
        The lightning_module.__call__ method must contain the ENTIRE forward pass logic.
        If there is any preprocessing (like flattening) it must be included in
        model.__call__ and not just in the lightning_module.forward method.
        The recommended practice is to implement all model logic in a separate nn.Module
        class and use that inside the LightningModule.
    """

    class Analyzer(lightning_module):
        """Analyzer wrapper that extends a LightningModule with analysis.

        This class dynamically inherits from the provided LightningModule and
        overrides the training_step to add:
        - Sample-wise gradient computation before each training step
        - Linearization probing for model analysis
        - Manual optimization control to support custom analysis workflows

        The wrapped module must have a 'criterion' attribute for loss
        computation.

        Attributes:
            sample_calc: Calculator for computing sample-wise gradients and
                metrics.
            linearizer: Linearizer for probing model linearization properties.
            delegate_optimization: Whether to delegate optimization to the
                wrapped model.
        """

        def __init__(
            self,
            sample_wise_engine=sample_wise_engine,
            disable_analyzer=disable_analyzer,
            log_metrics=log_metrics,
            opacus_strict=opacus_strict,
            analyze_every=analyze_every,
            analysis_schedule=analysis_schedule,
            cross_response_loader=cross_response_loader,
            **model_kwargs,
        ):
            super().__init__(**model_kwargs)

            # Store analyzer-specific parameters
            if sample_wise_engine not in ["opacus", "functorch"]:
                raise ValueError(
                    "sample_wise_engine must be either 'opacus' or 'functorch'"
                )

            if sample_wise_engine == "functorch" and opacus_strict:
                raise ValueError(
                    "opacus_strict=True is only valid when sample_wise_engine='opacus'. "
                    "Either set sample_wise_engine='opacus' or remove opacus_strict."
                )

            if analyze_every is not None and analyze_every < 1:
                raise ValueError("analyze_every must be a positive integer")

            if sample_wise_engine == "functorch":
                self.sample_calc = SamplewiseCalculatorFunctorch()
            elif sample_wise_engine == "opacus":
                self.sample_calc = SamplewiseCalculatorOpacus(strict=opacus_strict)

            # Initialize the linearizer
            self.linearizer = Linearizer()

            # Set up cross-response loader iterator
            self.cross_response_loader = cross_response_loader
            self._cross_response_iter = None
            if cross_response_loader is not None:
                self._cross_response_iter = iter(cross_response_loader)

            self.coupling_calc = CouplingCalculator()
            self.disable_analyzer = disable_analyzer
            self.log_metrics = log_metrics

            # Store scheduling options
            self.analyze_every = analyze_every
            self.analysis_schedule = analysis_schedule

            # Use manual optimization to control optimization steps
            if not self.automatic_optimization:
                warnings.warn(
                    "The wrapped model uses manual optimization. "
                    "Gradient Updates will be delegated to the the wrapped model's "
                    "training_step."
                )
                self.delegate_optimization = True
            else:
                self.delegate_optimization = False
            self.automatic_optimization = False  # We handle optimization manually

            # Check if model has criterion attribute
            if not hasattr(self, "criterion"):
                raise AttributeError(
                    "The wrapped model must have a 'criterion' attribute for loss "
                    "computation."
                )

        def training_step(self, batch, batch_idx):
            """Training step wrapper that adds sample-wise analysis.

            Performs analysis before and after the wrapped module's training
            step:
            1. Computes sample-wise gradients and metrics
            2. Probes linearization properties
            3. Executes the original training step
            4. Handles optimization (unless delegated to wrapped model)

            Args:
                batch: Training batch containing input data and labels.
                batch_idx: Index of the current batch.

            Returns:
                Output from the wrapped module's training_step.
            """
            # Initializing manual optimization
            opt = self.optimizers()
            opt.zero_grad()

            # BEFORE logic
            if not self.disable_analyzer:
                self._before_training_step(batch, batch_idx)

            # Original training step
            output = super().training_step(batch, batch_idx)
            if not self.delegate_optimization:
                # Backward pass
                self.manual_backward(output)
                # Optimizer step
                opt.step()

            # AFTER logic
            if not self.disable_analyzer:
                self._after_training_step(batch, batch_idx, output)

            return output

        def _should_analyze(self, step: int) -> bool:
            """Determine if analysis should run at the given step."""
            # If schedule provided, use it
            if self.analysis_schedule is not None:
                return self.analysis_schedule.should_analyze(step)
            # If analyze_every provided, check interval
            if self.analyze_every is not None:
                return step % self.analyze_every == 0
            # Default: analyze every step
            return True

        def _before_training_step(self, batch, batch_idx):
            """Hook executed before the wrapped training step.

            Computes analysis metrics including sample-wise gradients and
            linearization probes. Only runs if the scheduler determines
            this step should be analyzed.

            Args:
                batch: Training batch containing input data and labels.
                batch_idx: Index of the current batch.

            Returns:
                None
            """
            # Check if we should run analysis at this step
            if not self._should_analyze(self.global_step):
                return None

            x, y = batch

            # Get cross-response batch if available
            x2, y2 = None, None
            if self._cross_response_iter is not None:
                try:
                    x2, y2 = next(self._cross_response_iter)
                except StopIteration:
                    # Reset iterator and get first batch
                    self._cross_response_iter = iter(self.cross_response_loader)
                    x2, y2 = next(self._cross_response_iter)
                # Move to same device as training batch
                x2 = x2.to(x.device)
                y2 = y2.to(y.device)

            with BatchStatSnapshot(self.model, x):
                # Compute samplewise metrics
                samples_results = self.sample_calc.compute(
                    self.model,
                    self.criterion,
                    x,
                    y,
                )
                # Linearizer compute
                probe_results = self.linearizer.compute(
                    model=self.model,
                    criterion=self.criterion,
                    x1=x,
                    y1=y,
                    x2=x2,
                    y2=y2,
                )

                # Get "self" result for coupling calculation
                loss_self, _, delta_loss_self = probe_results["self"]

                # Compute coupling value (using self response)
                coupling_value = self.coupling_calc.calculate(
                    delta_loss=delta_loss_self,
                    chi_loss=samples_results["batch_grad_norms_loss"],
                    chi_net=samples_results["batch_grad_norms_network"],
                )

            # Log results with fixed metric names
            if self.log_metrics:
                self.log("chi_net", samples_results["batch_grad_norms_network"])
                self.log("chi_loss", samples_results["batch_grad_norms_loss"])
                self.log("coupling", coupling_value)
                self.log("batch_size", x.shape[0])
                self.log("analysis_step", self.global_step)

                # Log self response
                loss_self, perturbed_loss_self, delta_loss_self = probe_results["self"]
                self.log("loss", loss_self)
                self.log("grad_norm_squared", -delta_loss_self)

                # Log cross response if available
                if probe_results["cross"] is not None:
                    loss_cross, perturbed_loss_cross, delta_loss_cross = probe_results[
                        "cross"
                    ]
                    self.log("cross_loss", loss_cross)
                    self.log("cross_grad_dot_product", -delta_loss_cross)

                # Log window tracking info if using logarithmic schedule
                if self.analysis_schedule is not None:
                    window_info = self.analysis_schedule.get_window_info(
                        self.global_step
                    )
                    if window_info is not None:
                        self.log("window_id", window_info["window_id"])
                        self.log("window_center", window_info["window_center"])
                        self.log("window_width", window_info["window_width"])

            return None

        def _after_training_step(self, batch, batch_idx, output):
            """Hook executed after the wrapped training step.

            Placeholder for post-training step analysis logic.

            Args:
                batch: Training batch containing input data and labels.
                batch_idx: Index of the current batch.
                output: Output from the wrapped module's training_step.
            """
            pass

    return Analyzer(
        sample_wise_engine=sample_wise_engine,
        disable_analyzer=disable_analyzer,
        log_metrics=log_metrics,
        opacus_strict=opacus_strict,
        analyze_every=analyze_every,
        analysis_schedule=analysis_schedule,
        cross_response_loader=cross_response_loader,
        **model_kwargs,
    )
