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
    opacus_approximate_with_n: Optional[int] = None,
    analyze_every: Optional[int] = None,
    analysis_schedule: Optional[LogarithmicWindowSchedule] = None,
    cross_response: bool = False,
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
        opacus_approximate_with_n: If not None and using 'opacus' engine, use
            Hutchinson's trace estimator with n random projections instead of
            iterating over all output dimensions. This provides faster but
            approximate computation. Defaults to None (exact computation).
        analyze_every: If provided, run analysis every N steps (0, N, 2N, ...).
            If None and no analysis_schedule, runs every step.
        analysis_schedule: A LogarithmicWindowSchedule that defines which steps
            to analyze. Created via `logarithmic_windows()`. If provided,
            analysis runs only at the scheduled steps.
            If both analyze_every and analysis_schedule are provided, analysis_schedule
            takes precedence.
        cross_response: If True, enables cross-batch response analysis and assumes
            the training batch is a dict with 'train' and 'measure' keys.
            Defaults to False.
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
            opacus_approximate_with_n=opacus_approximate_with_n,
            analyze_every=analyze_every,
            analysis_schedule=analysis_schedule,
            cross_response=cross_response,
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

            if (
                sample_wise_engine == "functorch"
                and opacus_approximate_with_n is not None
            ):
                raise ValueError(
                    "opacus_approximate_with_n is only valid when sample_wise_engine='opacus'. "
                    "Either set sample_wise_engine='opacus' or remove opacus_approximate_with_n."
                )

            if analyze_every is not None and analyze_every < 1:
                raise ValueError("analyze_every must be a positive integer")

            if sample_wise_engine == "functorch":
                self.sample_calc = SamplewiseCalculatorFunctorch()
            elif sample_wise_engine == "opacus":
                self.sample_calc = SamplewiseCalculatorOpacus(
                    strict=opacus_strict, approximate_with_n=opacus_approximate_with_n
                )

            # Initialize the linearizer
            self.linearizer = Linearizer()

            # Set up cross-response loader iterator
            self.cross_response = cross_response

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
            batch_measure = None
            if self.cross_response:
                # Check if cross-response batch is provided
                if (
                    not isinstance(batch, dict)
                    or "train" not in batch
                    or "measure" not in batch
                ):
                    raise ValueError(
                        "When cross_response is True, the training batch must be a "
                        "dict with 'train' and 'measure' keys. "
                        "This can be achieved by using a CombinedLoader with mode='max_size_cycle'."
                    )
                batch_measure = batch["measure"]
                batch = batch["train"]

            # Initializing manual optimization
            opt = self.optimizers()
            opt.zero_grad()

            # BEFORE logic
            if not self.disable_analyzer:
                self._before_training_step(batch, batch_idx, batch_measure)

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

        def _before_training_step(self, batch, batch_idx, cross_response_batch=None):
            """Hook executed before the wrapped training step.

            Computes analysis metrics including sample-wise gradients and
            linearization probes. Only runs if the scheduler determines
            this step should be analyzed.

            Args:
                batch: Training batch containing input data and labels.
                batch_idx: Index of the current batch.
                cross_response_batch: Optional batch for cross-response analysis.

            Returns:
                None
            """
            # Check if we should run analysis at this step
            if not self._should_analyze(self.global_step):
                return None

            x, y = batch

            # Get cross-response batch if available
            x2, y2 = None, None
            if self.cross_response:
                x2, y2 = cross_response_batch

            samples_results = {}
            with BatchStatSnapshot(self.model, x):
                # Compute samplewise metrics for the training batch
                samples_results["self"] = self.sample_calc.compute(
                    self.model,
                    self.criterion,
                    x,
                    y,
                )
                # Compute samplewise metrics for the cross batch if available
                if x2 is not None and y2 is not None:
                    probe_results_cross_preliminary = self.sample_calc.compute(
                        self.model,
                        self.criterion,
                        x2,
                        y2,
                    )
                    samples_results["cross"] = self.sample_calc.compute_cross_metrics(
                        sample_wise_metrics_self=samples_results["self"],
                        sample_wise_metrics_cross=probe_results_cross_preliminary,
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
                chi_coup = self.coupling_calc.calculate(
                    delta_loss=delta_loss_self,
                    chi_loss=samples_results["self"]["batch_grad_norms_loss"],
                    chi_net=samples_results["self"]["batch_grad_norms_network"],
                )
                if self.cross_response and "cross" in probe_results:
                    # Optionally, compute coupling for cross response as well
                    loss_cross, _, delta_loss_cross = probe_results["cross"]
                    chi_coup_cross = self.coupling_calc.calculate(
                        delta_loss=delta_loss_cross,
                        chi_loss=samples_results["cross"]["batch_grad_norms_loss"],
                        chi_net=samples_results["cross"]["batch_grad_norms_network"],
                    )

            # Log results with fixed metric names
            if self.log_metrics:
                self._log_analysis_results(
                    prefix="",
                    samples_result=samples_results["self"],
                    probe_result=probe_results["self"],
                    chi_coup=chi_coup,
                    batch_size=x.shape[0],
                )

                # Log cross response if available
                if "cross" in samples_results and samples_results["cross"] is not None:
                    self._log_analysis_results(
                        prefix="cross_",
                        samples_result=samples_results["cross"],
                        probe_result=probe_results["cross"],
                        chi_coup=chi_coup_cross,
                        batch_size=x2.shape[0] if x2 is not None else 0,
                    )

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

        def _log_analysis_results(
            self,
            prefix: str,
            samples_result: dict,
            probe_result: tuple,
            chi_coup: Optional[float],
            batch_size: int,
        ):
            """Helper method to log analysis metrics with a given prefix."""
            # Log sample-wise metrics
            if "batch_grad_norms_network" in samples_result:
                self.log(f"{prefix}chi_net", samples_result["batch_grad_norms_network"])
            if "batch_grad_norms_loss" in samples_result:
                self.log(f"{prefix}chi_loss", samples_result["batch_grad_norms_loss"])

            # Log coupling if provided
            if chi_coup is not None:
                self.log(f"{prefix}chi_coup", chi_coup)

            self.log(f"{prefix}batch_size", batch_size)

            # Only log analysis_step once (usually with empty prefix)
            if prefix == "":
                self.log("analysis_step", self.global_step)

            # Log probe results (linearization)
            if probe_result is not None:
                loss, _, delta_loss = probe_result
                self.log(f"{prefix}loss", loss)

                # For cross response, we might want to name it differently or keep
                # consistent.
                # The original code used 'grad_norm_squared' for self and
                # 'cross_grad_dot_product' for cross.
                # We can standardize or keep the distinction based on prefix.
                metric_name = (
                    "grad_norm_squared" if prefix == "" else "grad_dot_product"
                )
                self.log(f"{prefix}{metric_name}", -delta_loss)

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
        opacus_approximate_with_n=opacus_approximate_with_n,
        analyze_every=analyze_every,
        analysis_schedule=analysis_schedule,
        cross_response=cross_response,
        **model_kwargs,
    )
