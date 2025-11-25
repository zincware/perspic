import warnings
from typing import Optional

import pytorch_lightning as pl

from perspic.calculator.linearizer import Linearizer
from perspic.calculator.samplewise import SamplewiseCalculatorFunctorch


def analyzer(
    lightning_module: pl.LightningModule,
    sample_wise_engine: Optional[str] = "functorch",
    disable_analyzer: bool = False,
    log_metrics: bool = True,
    linearizing_lrs: list[float] = [1e-3, 1e-5, 1e-7],
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
            Options: 'functorch' or 'opacus'. Defaults to 'functorch'.
        disable_analyzer: If True, wraps the module without adding analysis
            capabilities. Defaults to False. Mainly for testing purposes.
        log_metrics: If True, logs analysis metrics during training. Defaults to True.
        linearizing_lrs: List of learning rates for linear probing steps. Defaults to
            [1e-1, 1e-2, 1e-3, 1e-6].
        **model_kwargs: Additional keyword arguments passed to the
            LightningModule constructor.

    Returns:
        An initialized Analyzer instance that wraps the provided
        LightningModule.

    Raises:
        ValueError: If sample_wise_engine is not 'functorch' or 'opacus'.
        AttributeError: If the wrapped module doesn't have a 'criterion'
            attribute.
        NotImplementedError: If sample_wise_engine='opacus' is selected
            (not yet supported).

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
            linearizing_lrs=linearizing_lrs,
            **model_kwargs,
        ):
            super().__init__(**model_kwargs)

            # Store analyzer-specific parameters
            if sample_wise_engine not in ["opacus", "functorch"]:
                raise ValueError(
                    "sample_wise_engine must be either 'opacus' or 'functorch'"
                )

            if sample_wise_engine == "functorch":
                self.sample_calc = SamplewiseCalculatorFunctorch()
            elif sample_wise_engine == "opacus":
                raise NotImplementedError(
                    "sample_wise_engine='opacus' is not supported yet."
                )

            self.linearizer = Linearizer(eta_array=linearizing_lrs)
            self.disable_analyzer = disable_analyzer
            self.log_metrics = log_metrics

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

        def _before_training_step(self, batch, batch_idx):
            """Hook executed before the wrapped training step.

            Computes analysis metrics including sample-wise gradients and
            linearization probes.

            Args:
                batch: Training batch containing input data and labels.
                batch_idx: Index of the current batch.

            Returns:
                Tuple of (samples_results, probe_results) containing computed
                metrics.
            """
            x, y = batch

            # Compute samplewise metrics
            samples_results = self.sample_calc.compute(
                self.model,
                self.criterion,
                x,
                y,
            )

            # Linearizer probe
            probe_results = self.linearizer.probe_train_step(
                model=self.model,
                criterion=self.criterion,
                x=x,
                y=y,
            )

            # Log results
            if self.log_metrics:
                self.log(
                    "batch_grad_norms_network",
                    samples_results["batch_grad_norms_network"],
                )
                self.log(
                    "batch_grad_norms_loss",
                    samples_results["batch_grad_norms_loss"],
                )
                for eta, (loss, perturbed_loss) in probe_results.items():
                    eta_str = f"{eta:.0e}"
                    self.log(f"lin_loss_before_eta_{eta_str}", loss)
                    if perturbed_loss is not None:
                        self.log(f"lin_loss_after_eta_{eta_str}", perturbed_loss)
                self.log("loss_value", probe_results[list(probe_results.keys())[0]][0])
                self.log("actual_batch_size", x.shape[0])

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

    return Analyzer(sample_wise_engine=sample_wise_engine, **model_kwargs)
