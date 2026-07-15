import math
import warnings
from typing import Optional, Union

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
    micro_batch_size: Optional[int] = None,
    effective_batch_size: Optional[int] = None,
    measure_dataloader: Optional[torch.utils.data.DataLoader] = None,
    measure_batch_size: Optional[Union[int, list[int]]] = None,
    measure_subset_seed: Optional[int] = None,
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
        cross_response: If True, enables cross-batch response
            analysis and assumes the training batch is a dict
            with 'train' and 'measure' keys. Defaults to False.
        micro_batch_size: The actual micro-batch size used by the
            DataLoader. Required when effective_batch_size is
            set. Can be provided alone (no accumulation).
        effective_batch_size: The desired simulated batch size
            achieved through gradient accumulation. Must be
            divisible by micro_batch_size. When set, the optimizer
            step is only performed every
            (effective_batch_size // micro_batch_size) micro-batches.
        measure_dataloader: An independent DataLoader supplying the
            measurement ("cross") batch, decoupled from the training
            batch/accumulation. When set, the analyzer owns a
            persistent iterator over this loader and the training
            batch is expected to be a plain (x, y) tuple (not a
            CombinedLoader dict). The measure micro-batch size is
            inferred from measure_dataloader.batch_size; use
            drop_last=True (or MultiEpochsDataLoader) so every pulled
            micro-batch is full. Setting this activates the
            independent measure-response path and implies
            cross-response-style analysis regardless of
            cross_response.
        measure_batch_size: The desired measurement batch size(s),
            achieved through measure-side gradient accumulation when
            larger than measure_dataloader.batch_size. Accepts a
            single int or a list[int] to sweep multiple sizes in one
            analysis step (each swept size is logged with a
            "@bs{S}" suffix). Each value must be >= and divisible by
            measure_dataloader.batch_size. Defaults to
            measure_dataloader.batch_size (no measure accumulation).
            Only valid together with measure_dataloader.
        measure_subset_seed: Seed for the random generator used to
            draw reproducible subsets of the measure pool when
            sweeping multiple measure_batch_size values. Only valid
            together with measure_dataloader.
        **model_kwargs: Additional keyword arguments passed to
            the LightningModule constructor.

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
            micro_batch_size=micro_batch_size,
            effective_batch_size=effective_batch_size,
            measure_dataloader=measure_dataloader,
            measure_batch_size=measure_batch_size,
            measure_subset_seed=measure_subset_seed,
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

            # Gradient accumulation setup
            self.micro_batch_size = micro_batch_size
            self.effective_batch_size = effective_batch_size

            if effective_batch_size is not None and micro_batch_size is None:
                raise ValueError(
                    "micro_batch_size must be specified when "
                    "effective_batch_size is set."
                )

            if micro_batch_size is not None and effective_batch_size is not None:
                if effective_batch_size < micro_batch_size:
                    raise ValueError(
                        f"effective_batch_size "
                        f"({effective_batch_size}) must be "
                        f">= micro_batch_size ({micro_batch_size})."
                    )
                if effective_batch_size % micro_batch_size != 0:
                    raise ValueError(
                        f"effective_batch_size "
                        f"({effective_batch_size}) must be "
                        f"divisible by micro_batch_size "
                        f"({micro_batch_size})."
                    )
                self.accumulation_steps = effective_batch_size // micro_batch_size
            else:
                self.accumulation_steps = 1

            if self.accumulation_steps > 1 and self.delegate_optimization:
                raise ValueError(
                    "Gradient accumulation is not supported "
                    "when the wrapped model uses manual "
                    "optimization (delegate_optimization=True)."
                )

            # --- Independent measure data source (cross-response, decoupled sizing) ---
            self._measure_dataloader = measure_dataloader
            self._independent_measure = measure_dataloader is not None

            if not self._independent_measure and (
                measure_batch_size is not None or measure_subset_seed is not None
            ):
                raise ValueError(
                    "measure_batch_size and measure_subset_seed are only "
                    "valid together with measure_dataloader."
                )

            if self._independent_measure:
                measure_micro = getattr(measure_dataloader, "batch_size", None)
                if measure_micro is None:
                    raise ValueError(
                        "measure_dataloader must have an integer batch_size "
                        "(its batch_size attribute is None). Provide a "
                        "DataLoader whose batch_size divides every "
                        "measure_batch_size."
                    )
                self._measure_micro_batch_size = measure_micro

                if measure_batch_size is None:
                    sizes = [measure_micro]
                elif isinstance(measure_batch_size, int):
                    sizes = [measure_batch_size]
                else:
                    sizes = list(measure_batch_size)
                    if len(sizes) == 0:
                        raise ValueError("measure_batch_size list must be non-empty.")

                for s in sizes:
                    if not isinstance(s, int):
                        raise ValueError(
                            f"measure_batch_size entries must be integers, "
                            f"got {type(s)}."
                        )
                    if s < 1:
                        raise ValueError(
                            f"measure_batch_size entries must be positive, " f"got {s}."
                        )
                    if s > measure_micro and s % measure_micro != 0:
                        raise ValueError(
                            f"measure_batch_size ({s}) is larger than the "
                            f"measure_dataloader batch_size ({measure_micro}) "
                            f"— the maximum single-pass size — and must then "
                            f"be an exact multiple of it, to be measured via "
                            f"gradient accumulation. Sizes <= {measure_micro} "
                            f"need no such constraint (they run as a single "
                            f"direct pass)."
                        )
                self._measure_batch_sizes = sizes

                if len(sizes) > 1 and analysis_schedule is None:
                    warnings.warn(
                        "A measure batch-size sweep (measure_batch_size "
                        f"list of length {len(sizes)}) without a "
                        "logarithmic analysis_schedule runs the full sweep "
                        "at EVERY analyzed step and is very expensive. Pass "
                        "analysis_schedule=logarithmic_windows(...) to "
                        "restrict analysis to logarithmically spaced steps."
                    )

                self._measure_gen = torch.Generator()
                if measure_subset_seed is not None:
                    self._measure_gen.manual_seed(measure_subset_seed)
                self._measure_iter = None
            else:
                self._measure_micro_batch_size = None
                self._measure_batch_sizes = None
                self._measure_gen = None
                self._measure_iter = None

            self._accumulation_count = 0
            self._optimizer_step_count = 0

            # Analysis accumulation buffers
            self._accum_chi_net = []
            self._accum_chi_loss = []
            self._accum_cross_chi_net = []
            self._accum_cross_chi_loss = []
            self._accum_grad_train = None
            self._accum_grad_measure = None
            self._accum_train_loss = 0.0
            self._accum_measure_loss = 0.0
            self._accum_step_losses = []  # micro-batch losses for per-opt-step logging
            # Track whether analysis is active for this cycle
            self._analysis_active = False

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
            if self.cross_response and not self._independent_measure:
                # Unpack batch if provided as tuple (batch, batch_idx, dataloader_idx)
                if type(batch) is tuple and len(batch) == 3:
                    batch, _batch_idx, dataloader_idx = batch
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

            # Zero gradients only at start of accumulation cycle
            if self._accumulation_count == 0:
                opt.zero_grad()

            # BEFORE logic
            if not self.disable_analyzer:
                self._before_training_step(batch, batch_idx, batch_measure)

            # Original training step
            output = super().training_step(batch, batch_idx)
            if not self.delegate_optimization:
                # Scale loss for gradient accumulation
                scaled_output = output / self.accumulation_steps
                self.manual_backward(scaled_output)

                self._accumulation_count += 1

                if self.accumulation_steps > 1:
                    self._accum_step_losses.append(output.detach())
                    # Tag every micro-batch with its cycle's effective_step so
                    # groupby(effective_step).mean() in the plot averages exactly
                    # the K micro-batches of that cycle (no Lightning forward-fill
                    # ambiguity). opt.step() has not fired yet, so +1 gives the
                    # current cycle number.
                    self.log(
                        "effective_step",
                        float(self._optimizer_step_count + 1),
                        on_step=True,
                        on_epoch=False,
                    )

                # Step optimizer only at end of accumulation cycle
                if self._accumulation_count >= self.accumulation_steps:
                    opt.step()
                    self._optimizer_step_count += 1
                    self._accumulation_count = 0

                    if self.accumulation_steps > 1 and self._accum_step_losses:
                        self._accum_step_losses.clear()

                    # Step schedulers with interval='step'
                    if self._trainer is not None and self.trainer.lr_scheduler_configs:
                        for config in self.trainer.lr_scheduler_configs:
                            if config.interval == "step":
                                config.scheduler.step()

            # AFTER logic
            if not self.disable_analyzer:
                self._after_training_step(batch, batch_idx, output)

            return output

        def on_train_epoch_end(self):
            """Hook executed at the end of the training epoch."""
            # Step schedulers with interval='epoch'
            if (
                not self.delegate_optimization
                and self._trainer is not None
                and self.trainer.lr_scheduler_configs
            ):
                for config in self.trainer.lr_scheduler_configs:
                    if config.interval == "epoch":
                        config.scheduler.step()

            super().on_train_epoch_end()

        @property
        def effective_step(self):
            """Return the effective optimizer step count."""
            if self.delegate_optimization:
                return self.global_step
            return self._optimizer_step_count

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
            linearization probes. When gradient accumulation is active,
            metrics are accumulated across micro-batches and only logged
            after the full accumulation cycle.

            Args:
                batch: Training batch containing input data and labels.
                batch_idx: Index of the current batch.
                cross_response_batch: Optional batch for cross-response analysis.

            Returns:
                None
            """
            if self.accumulation_steps == 1:
                return self._analyze_single_step(batch, batch_idx, cross_response_batch)
            else:
                return self._analyze_accumulated_step(
                    batch, batch_idx, cross_response_batch
                )

        def _analyze_single_step(self, batch, batch_idx, cross_response_batch=None):
            """Run analysis for a single step (no accumulation)."""
            if not self._should_analyze(self.effective_step):
                return None

            x, y = batch
            # Get cross-response batch if applicable
            x2, y2 = None, None
            if self.cross_response:
                x2, y2 = cross_response_batch

            samples_results = {}
            with BatchStatSnapshot(self.model, x):
                # Compute sample-wise metrics and self response
                samples_results["self"] = self.sample_calc.compute(
                    self.model,
                    self.criterion,
                    x,
                    y,
                )
                # Compute sample-wise metrics and cross response if applicable
                if x2 is not None and y2 is not None:
                    cross_preliminary = self.sample_calc.compute(
                        self.model,
                        self.criterion,
                        x2,
                        y2,
                    )
                    samples_results["cross"] = self.sample_calc.compute_cross_metrics(
                        sample_wise_metrics_self=samples_results["self"],
                        sample_wise_metrics_cross=cross_preliminary,
                    )
                # Linearizer probe
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
                chi_coup = self.coupling_calc.calculate(
                    delta_loss=delta_loss_self,
                    chi_loss=samples_results["self"]["batch_grad_norms_loss"],
                    chi_net=samples_results["self"]["batch_grad_norms_network"],
                )
                chi_coup_cross = None
                if self.cross_response and "cross" in probe_results:
                    _, _, delta_loss_cross = probe_results["cross"]
                    chi_coup_cross = self.coupling_calc.calculate(
                        delta_loss=delta_loss_cross,
                        chi_loss=samples_results["cross"]["batch_grad_norms_loss"],
                        chi_net=samples_results["cross"]["batch_grad_norms_network"],
                    )

                # Capture the train gradient for the independent measure-response
                # path. Linearizer.compute() zeroes grads internally and discards
                # its own gradient, so we recompute it here (K_train=1) rather than
                # reaching into the linearizer's internals.
                grad_train_mean = None
                if self._independent_measure:
                    self.model.zero_grad()
                    loss_t = self.criterion(self.model(x), y)
                    loss_t.backward()
                    grad_train_mean = [
                        p.grad.clone() if p.grad is not None else None
                        for p in self.model.parameters()
                    ]
                    self.model.zero_grad()
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
                        self.effective_step
                    )
                    if window_info is not None:
                        self.log("window_id", window_info["window_id"])
                        self.log("window_center", window_info["window_center"])
                        self.log("window_width", window_info["window_width"])

                if self._independent_measure:
                    self._measure_response(
                        grad_train_mean=grad_train_mean,
                        self_chi_metrics=samples_results["self"],
                    )

            return None

        def _analyze_accumulated_step(
            self, batch, batch_idx, cross_response_batch=None
        ):
            """Run analysis with gradient accumulation across micro-batches.

            On each micro-batch: accumulate sample-wise metrics and linearizer
            gradients. On the last micro-batch of the cycle: finalize, log, clear.
            """
            # On first micro-batch of cycle, decide whether to analyze
            if self._accumulation_count == 0:
                self._analysis_active = self._should_analyze(self.effective_step)
                if self._analysis_active:
                    self._clear_accumulation_buffers()

            if not self._analysis_active:
                return None

            x, y = batch
            x2, y2 = None, None
            if self.cross_response:
                x2, y2 = cross_response_batch

            # Save training grads before any analysis backward/zero_grad calls.
            # sample_calc.compute and _accumulate_linearizer_grads both call
            # model.zero_grad() internally; restoring here ensures the training
            # accumulation loop sees unmodified gradients after this hook.
            saved_grads = [
                p.grad.clone() if p.grad is not None else None
                for p in self.model.parameters()
            ]

            with BatchStatSnapshot(self.model, x):
                # Accumulate sample-wise metrics
                self_metrics = self.sample_calc.compute(
                    self.model,
                    self.criterion,
                    x,
                    y,
                )
                self._accum_chi_net.append(self_metrics["batch_grad_norms_network"])
                self._accum_chi_loss.append(self_metrics["batch_grad_norms_loss"])

                if x2 is not None and y2 is not None:
                    cross_preliminary = self.sample_calc.compute(
                        self.model,
                        self.criterion,
                        x2,
                        y2,
                    )
                    cross_metrics = self.sample_calc.compute_cross_metrics(
                        sample_wise_metrics_self=self_metrics,
                        sample_wise_metrics_cross=cross_preliminary,
                    )
                    self._accum_cross_chi_net.append(
                        cross_metrics["batch_grad_norms_network"]
                    )
                    self._accum_cross_chi_loss.append(
                        cross_metrics["batch_grad_norms_loss"]
                    )

                # Accumulate linearizer gradients (train side)
                self._accumulate_linearizer_grads(x, y, is_train=True)
                # Accumulate linearizer gradients (measure side). This legacy
                # path ties the measure batch to the train accumulation cycle
                # (CombinedLoader "measure" key, size = K_train * micro). For
                # an independently sized (and independently accumulated)
                # measure batch, use measure_dataloader/measure_batch_size
                # instead (see _measure_response), which accumulates and
                # combines the measure side separately with its own K.
                if x2 is not None and y2 is not None:
                    self._accumulate_linearizer_grads(x2, y2, is_train=False)

            # Restore training grads clobbered by analysis backward passes
            for p, s in zip(self.model.parameters(), saved_grads):
                p.grad = s

            # On last micro-batch: finalize and log
            is_last = self._accumulation_count == self.accumulation_steps - 1
            if is_last:
                self._finalize_accumulated_analysis(x, x2)

            return None

        def _accumulate_linearizer_grads(self, x, y, is_train=True):
            """Forward+backward on a micro-batch and add grads to accumulator.

            Must be called inside a BatchStatSnapshot context (caller's
            responsibility). Training grads are saved/restored by the caller
            (_analyze_accumulated_step) around the full analysis block.
            """
            self.model.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()

            loss_val = loss.detach().item()
            if is_train:
                self._accum_train_loss += loss_val
                if self._accum_grad_train is None:
                    self._accum_grad_train = [
                        p.grad.clone() if p.grad is not None else None
                        for p in self.model.parameters()
                    ]
                else:
                    for acc, p in zip(self._accum_grad_train, self.model.parameters()):
                        if acc is not None and p.grad is not None:
                            acc.add_(p.grad)
            else:
                self._accum_measure_loss += loss_val
                if self._accum_grad_measure is None:
                    self._accum_grad_measure = [
                        p.grad.clone() if p.grad is not None else None
                        for p in self.model.parameters()
                    ]
                else:
                    for acc, p in zip(
                        self._accum_grad_measure, self.model.parameters()
                    ):
                        if acc is not None and p.grad is not None:
                            acc.add_(p.grad)

        def _finalize_accumulated_analysis(self, x, x2):
            """Combine accumulated metrics and log results."""
            K = self.accumulation_steps
            B = x.shape[0]

            # Combine sample-wise metrics.
            # Both chi_net and chi_loss are computed with normalize=True, which
            # makes them extensive in the batch size via a 1/B or *B factor
            # derived from mean-reduced loss. For an effective batch N=K*B, the
            # correct aggregate for both quantities is the mean across micro-batches.
            chi_net_eff = sum(self._accum_chi_net) / K
            chi_loss_eff = sum(self._accum_chi_loss) / K

            samples_result_self = {
                "batch_grad_norms_network": chi_net_eff,
                "batch_grad_norms_loss": chi_loss_eff,
            }

            # Compute self linearizer result from accumulated grads
            grad_norm_sq = sum(
                (g**2).sum().item() for g in self._accum_grad_train if g is not None
            ) / (K**2)

            avg_train_loss = self._accum_train_loss / K
            delta_loss_self = -grad_norm_sq
            probe_result_self = (
                avg_train_loss,
                avg_train_loss + delta_loss_self,
                delta_loss_self,
            )

            chi_coup = self.coupling_calc.calculate(
                delta_loss=delta_loss_self,
                chi_loss=chi_loss_eff,
                chi_net=chi_net_eff,
            )

            # Cross response
            samples_result_cross = None
            probe_result_cross = None
            chi_coup_cross = None
            if self._accum_grad_measure is not None:
                chi_net_cross_eff = sum(self._accum_cross_chi_net) / K
                chi_loss_cross_eff = sum(self._accum_cross_chi_loss) / K
                samples_result_cross = {
                    "batch_grad_norms_network": chi_net_cross_eff,
                    "batch_grad_norms_loss": chi_loss_cross_eff,
                }

                cross_dot = sum(
                    (g1 * g2).sum().item()
                    for g1, g2 in zip(
                        self._accum_grad_train,
                        self._accum_grad_measure,
                    )
                    if g1 is not None and g2 is not None
                ) / (K**2)

                avg_measure_loss = self._accum_measure_loss / K
                delta_loss_cross = -cross_dot
                probe_result_cross = (
                    avg_measure_loss,
                    avg_measure_loss + delta_loss_cross,
                    delta_loss_cross,
                )
                chi_coup_cross = self.coupling_calc.calculate(
                    delta_loss=delta_loss_cross,
                    chi_loss=chi_loss_cross_eff,
                    chi_net=chi_net_cross_eff,
                )

            # Log results
            if self.log_metrics:
                self._log_analysis_results(
                    prefix="",
                    samples_result=samples_result_self,
                    probe_result=probe_result_self,
                    chi_coup=chi_coup,
                    batch_size=B,
                )
                if samples_result_cross is not None:
                    self._log_analysis_results(
                        prefix="cross_",
                        samples_result=samples_result_cross,
                        probe_result=probe_result_cross,
                        chi_coup=chi_coup_cross,
                        batch_size=x2.shape[0] if x2 is not None else 0,
                    )
                if self.analysis_schedule is not None:
                    window_info = self.analysis_schedule.get_window_info(
                        self.effective_step
                    )
                    if window_info is not None:
                        self.log("window_id", window_info["window_id"])
                        self.log("window_center", window_info["window_center"])
                        self.log("window_width", window_info["window_width"])

                if self._independent_measure:
                    grad_train_mean = [
                        g / K if g is not None else None for g in self._accum_grad_train
                    ]
                    self._measure_response(
                        grad_train_mean=grad_train_mean,
                        self_chi_metrics=samples_result_self,
                    )

            self._clear_accumulation_buffers()

        def _clear_accumulation_buffers(self):
            """Reset all accumulation buffers."""
            self._accum_chi_net.clear()
            self._accum_chi_loss.clear()
            self._accum_cross_chi_net.clear()
            self._accum_cross_chi_loss.clear()
            self._accum_grad_train = None
            self._accum_grad_measure = None
            self._accum_train_loss = 0.0
            self._accum_measure_loss = 0.0
            self._accum_step_losses.clear()

        def _next_measure_micro_batch(self):
            """Pull one measure micro-batch from the persistent iterator.

            The iterator is created lazily on first use and refilled on
            exhaustion so a finite measure_dataloader cycles indefinitely
            (a MultiEpochsDataLoader is already infinite and simply keeps
            yielding). Returns tensors moved to self.device.
            """
            if self._measure_iter is None:
                self._measure_iter = iter(self._measure_dataloader)
            try:
                xb, yb = next(self._measure_iter)
            except StopIteration:
                self._measure_iter = iter(self._measure_dataloader)
                xb, yb = next(self._measure_iter)

            micro = self._measure_micro_batch_size
            if xb.shape[0] != micro:
                raise ValueError(
                    f"measure_dataloader yielded a micro-batch of size "
                    f"{xb.shape[0]}, expected {micro}. Use drop_last=True "
                    f"(or a dataset size divisible by batch_size) so every "
                    f"measure micro-batch is full."
                )
            return xb.to(self.device), yb.to(self.device)

        def _measure_response(self, grad_train_mean, self_chi_metrics):
            """Compute and log cross-response metrics against an independent
            measure batch, decoupled from the training accumulation.

            Gathers a single pool of size max(measure_batch_size) from the
            persistent measure iterator, then processes every requested
            measure batch size largest-to-smallest: the largest uses the
            whole pool, each smaller size uses a seed-fixable random subset
            of that same pool (so subset draws are reproducible given
            measure_subset_seed). For each size, the measure gradient and
            per-sample chi metrics are accumulated over its own micro-batch
            chunks (measure-side gradient accumulation), then combined and
            logged with a "@bs{S}" suffix when sweeping multiple sizes.

            Args:
                grad_train_mean: List aligned with self.model.parameters(),
                    the mean training gradient (accumulated grad / K_train).
                self_chi_metrics: Dict with "batch_grad_norms_network" and
                    "batch_grad_norms_loss", the aggregated train self chi
                    metrics used as the "self" side of the cross metric.

            Note:
                Runs its own forward/backward passes; saves and restores
                self.model gradients so the surrounding (possibly partially
                accumulated) training gradient is never corrupted.
            """
            saved_grads = [
                p.grad.clone() if p.grad is not None else None
                for p in self.model.parameters()
            ]

            sizes = sorted(self._measure_batch_sizes, reverse=True)
            S_max = sizes[0]
            micro = self._measure_micro_batch_size
            sweep = len(self._measure_batch_sizes) > 1

            # Gather ONE pool of S_max samples from the persistent iterator.
            # S_max need not be a multiple of micro (e.g. every swept size is
            # below the max single-pass size), so pull enough micro-batches
            # to cover it and slice down to exactly S_max samples.
            pool_x, pool_y = [], []
            for _ in range(math.ceil(S_max / micro)):
                xb, yb = self._next_measure_micro_batch()
                pool_x.append(xb)
                pool_y.append(yb)
            x_pool = torch.cat(pool_x, dim=0)[:S_max]
            y_pool = torch.cat(pool_y, dim=0)[:S_max]

            for S in sizes:
                if S == S_max:
                    idx = torch.arange(S_max, device=x_pool.device)
                else:
                    # Seed-fixable random subset of the same pool, drawn
                    # after larger sizes so the sweep is reproducible given
                    # measure_subset_seed regardless of how many sizes ran.
                    perm = torch.randperm(S_max, generator=self._measure_gen)
                    idx = perm[:S].to(x_pool.device)
                x_sel, y_sel = x_pool[idx], y_pool[idx]

                # chunk_size = min(S, micro) unifies both regimes: when
                # S <= micro this gives chunk_size=S, K_measure=1 (a single
                # direct pass using less than the max single-pass capacity,
                # no accumulation); when S > micro this gives
                # chunk_size=micro, K_measure=S // micro (accumulated passes
                # of the max single-pass size — exact, since S > micro must
                # be a multiple of micro per __init__ validation).
                chunk_size = min(S, micro)
                K_measure = S // chunk_size
                grad_measure_acc = None
                measure_loss_sum = 0.0
                chi_net_chunks, chi_loss_chunks = [], []

                for k in range(K_measure):
                    xc = x_sel[k * chunk_size : (k + 1) * chunk_size]
                    yc = y_sel[k * chunk_size : (k + 1) * chunk_size]
                    with BatchStatSnapshot(self.model, xc):
                        m = self.sample_calc.compute(self.model, self.criterion, xc, yc)
                        chi_net_chunks.append(m["batch_grad_norms_network"])
                        chi_loss_chunks.append(m["batch_grad_norms_loss"])

                        self.model.zero_grad()
                        loss = self.criterion(self.model(xc), yc)
                        loss.backward()
                        measure_loss_sum += loss.detach().item()
                        if grad_measure_acc is None:
                            grad_measure_acc = [
                                p.grad.clone() if p.grad is not None else None
                                for p in self.model.parameters()
                            ]
                        else:
                            for acc, p in zip(
                                grad_measure_acc, self.model.parameters()
                            ):
                                if acc is not None and p.grad is not None:
                                    acc.add_(p.grad)

                measure_chi = {
                    "batch_grad_norms_network": sum(chi_net_chunks) / K_measure,
                    "batch_grad_norms_loss": sum(chi_loss_chunks) / K_measure,
                }
                cross_metrics = self.sample_calc.compute_cross_metrics(
                    sample_wise_metrics_self=self_chi_metrics,
                    sample_wise_metrics_cross=measure_chi,
                )

                grad_measure_mean = [
                    g / K_measure if g is not None else None for g in grad_measure_acc
                ]
                cross_dot = sum(
                    (g1 * g2).sum().item()
                    for g1, g2 in zip(grad_train_mean, grad_measure_mean)
                    if g1 is not None and g2 is not None
                )
                avg_measure_loss = measure_loss_sum / K_measure
                delta_loss_cross = -cross_dot
                probe_result_cross = (
                    avg_measure_loss,
                    avg_measure_loss + delta_loss_cross,
                    delta_loss_cross,
                )
                chi_coup_cross = self.coupling_calc.calculate(
                    delta_loss=delta_loss_cross,
                    chi_loss=cross_metrics["batch_grad_norms_loss"],
                    chi_net=cross_metrics["batch_grad_norms_network"],
                )

                if self.log_metrics:
                    suffix = f"@bs{S}" if sweep else ""
                    self._log_analysis_results(
                        prefix="cross_",
                        samples_result=cross_metrics,
                        probe_result=probe_result_cross,
                        chi_coup=chi_coup_cross,
                        batch_size=S,
                        suffix=suffix,
                        log_effective_batch_size=False,
                    )

            for p, s in zip(self.model.parameters(), saved_grads):
                p.grad = s

        def _log_analysis_results(
            self,
            prefix: str,
            samples_result: dict,
            probe_result: tuple,
            chi_coup: Optional[float],
            batch_size: int,
            suffix: str = "",
            log_effective_batch_size: Optional[bool] = None,
        ):
            """Helper method to log analysis metrics with a given prefix.

            Args:
                suffix: Appended to every logged key. Used to disambiguate
                    swept measure batch sizes, e.g. "@bs2000".
                log_effective_batch_size: Whether to additionally log
                    "{prefix}effective_batch_size{suffix}" as
                    batch_size * accumulation_steps. Defaults to
                    accumulation_steps > 1 (existing behavior). Independent
                    measure batches pass False explicitly since the train
                    accumulation_steps multiplier has no meaning for them
                    (batch_size is already the full measure size).
            """
            if log_effective_batch_size is None:
                log_effective_batch_size = self.accumulation_steps > 1

            # Log sample-wise metrics
            if "batch_grad_norms_network" in samples_result:
                self.log(
                    f"{prefix}chi_net{suffix}",
                    samples_result["batch_grad_norms_network"],
                )
            if "batch_grad_norms_loss" in samples_result:
                self.log(
                    f"{prefix}chi_loss{suffix}",
                    samples_result["batch_grad_norms_loss"],
                )

            # Log coupling if provided
            if chi_coup is not None:
                self.log(f"{prefix}chi_coup{suffix}", chi_coup)

            self.log(f"{prefix}batch_size{suffix}", batch_size)
            if log_effective_batch_size:
                self.log(
                    f"{prefix}effective_batch_size{suffix}",
                    batch_size * self.accumulation_steps,
                )

            # Only log analysis_step once (usually with empty prefix/suffix)
            if prefix == "" and suffix == "":
                self.log("analysis_step", self.effective_step)

            # Log probe results (linearization)
            if probe_result is not None:
                loss, _, delta_loss = probe_result
                self.log(f"{prefix}loss{suffix}", loss)

                # For cross response, we might want to name it differently or keep
                # consistent.
                # The original code used 'grad_norm_squared' for self and
                # 'cross_grad_dot_product' for cross.
                # We can standardize or keep the distinction based on prefix.
                metric_name = (
                    "grad_norm_squared" if prefix == "" else "grad_dot_product"
                )
                self.log(f"{prefix}{metric_name}{suffix}", -delta_loss)

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
        micro_batch_size=micro_batch_size,
        effective_batch_size=effective_batch_size,
        measure_dataloader=measure_dataloader,
        measure_batch_size=measure_batch_size,
        measure_subset_seed=measure_subset_seed,
        **model_kwargs,
    )
