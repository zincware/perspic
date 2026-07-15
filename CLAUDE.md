# perspic

A tool to study neural network training dynamics. Package `perspic`, v0.0.1, authors Konstantin Nikolaou and Jonas Scheunemann, requires Python >=3.11. Built on PyTorch and PyTorch Lightning.

## Overview

perspic wraps a user's `pytorch_lightning.LightningModule` with an `analyzer()` factory, transparently instrumenting training to compute per-sample gradient-norm and linear-response metrics — quantities used to study training dynamics (gradient coupling, sensitivity, collective variables) — without requiring the user to modify their model code. The wrapped module trains normally under a standard `pytorch_lightning.Trainer`; the analysis runs and logs metrics (`chi_net`, `chi_loss`, `chi_coup`, `grad_norm_squared`, `loss`, `batch_size`, ...) alongside it.

## Architecture

`perspic/analyzer.py` — `analyzer(lightning_module, sample_wise_engine="opacus", disable_analyzer=False, log_metrics=True, opacus_strict=False, opacus_approximate_with_n=None, analyze_every=None, analysis_schedule=None, cross_response=False, micro_batch_size=None, effective_batch_size=None, measure_dataloader=None, measure_batch_size=None, measure_subset_seed=None, **model_kwargs)`. A factory function that dynamically subclasses the given `LightningModule` into an `Analyzer` class. `training_step` is overridden to: run pre-step analysis (`_before_training_step`) → delegate to the wrapped module's own `training_step` → manual backward and optimizer step → step any `interval='step'` LR schedulers → log results. `on_train_epoch_end` steps `interval='epoch'` LR schedulers. The wrapped module must define a `criterion` attribute. Supports `cross_response=True`, which expects a dict batch `{"train": ..., "measure": ...}` (via `CombinedLoader`) to additionally measure the model's linear response against a held-out batch, and sparse analysis scheduling via `analyze_every` or `analysis_schedule`.

`micro_batch_size`/`effective_batch_size` enable train-side gradient accumulation: the DataLoader yields `micro_batch_size` micro-batches, and the optimizer steps only every `effective_batch_size // micro_batch_size` of them, with per-sample gradient-norm metrics accumulated across the cycle and combined (via the correct extensive/intensive scaling — see `examples/batch_accumulation_notes.md`) before logging.

`measure_dataloader`/`measure_batch_size`/`measure_subset_seed` give the cross-response measurement side an **independent** batch size, decoupled from the train/`CombinedLoader` path above: the analyzer holds a persistent iterator over `measure_dataloader` and, on each analyzed step, gathers a pool of `max(measure_batch_size)` samples. `measure_batch_size` accepts an int or a `list[int]` to sweep multiple sizes per step (largest processed first; smaller sizes are seed-fixable random subsets of the same pool via `measure_subset_seed`). `measure_dataloader.batch_size` is the maximum single-pass batch size; each swept size `S <= measure_dataloader.batch_size` is measured with a single direct pass (no accumulation), while each `S >` it must be an exact multiple and is measured via gradient accumulation (`S // measure_dataloader.batch_size` passes combined into one measurement) — logging `cross_*` metrics with a `@bs{S}` suffix per swept size. See `examples/measurement_batch_sweep.md` for the full design and usage.

`perspic/calculator/` — the analysis engines used by `analyzer()`:
- `coupling.py` — `CouplingCalculator`: computes the coupling value `chi_coup = ||grad_L||^2 / (chi_loss * chi_net)`.
- `linearizer.py` — `Linearizer`: computes the exact first-order linear response of the loss via gradient dot products. `compute(model, criterion, x1, y1, x2=None, y2=None)` returns `(loss, perturbed_loss, delta_loss)`, with an optional cross term against a second batch.
- `samplewise.py` — `SamplewiseCalculator`: abstract base class defining the interface for per-sample gradient-norm calculators (`compute`, network- and loss-level norm helpers), plus shared helpers.
- `samplewise_functorch.py` — `SamplewiseCalculatorFunctorch`: per-sample gradient norms via `torch.func` (`vmap` + `jacrev`).
- `samplewise_opacus.py` — `SamplewiseCalculatorOpacus`: per-sample gradient norms via Opacus ghost clipping (`GradSampleModuleFastGradientClipping`), with custom BatchNorm grad/norm samplers for eval-mode frozen stats and an optional Hutchinson trace-estimator approximation (`approximate_with_n`) for the network-level norm.

`perspic/logger.py` — `LogarithmicWindowSchedule` (dataclass) and `logarithmic_windows(max_steps, points_per_decade=10, base_window=5, adaptive_scale=0.0)`, for building logarithmically-spaced step schedules so analysis can run sparsely over long training runs (pass as `analyzer(..., analysis_schedule=...)`).

`perspic/utils.py` — `BatchStatSnapshot`, a context manager that freezes BatchNorm running stats (with Bessel's correction) so per-sample `vmap` gradients match a train-mode forward pass; `MultiEpochsDataLoader` and `RepeatSampler`, a `DataLoader` subclass that reuses its workers/iterator across epochs.

Class design relies on: ABCs for the calculator hierarchy, a dataclass for the logging schedule, and a factory-returns-dynamic-subclass pattern for `analyzer()` (it builds `class Analyzer(lightning_module): ...` at call time rather than requiring users to subclass anything themselves).

## Public API

`from perspic import ...`:
- `analyzer` — the primary entry point; wraps and instantiates an `Analyzer` from a `LightningModule`.
- `Linearizer` — standalone linear-response calculator.
- `SamplewiseCalculatorFunctorch`, `SamplewiseCalculatorOpacus` — standalone per-sample gradient-norm calculators, also selectable inside `analyzer()` via `sample_wise_engine`.
- `LogarithmicWindowSchedule`, `logarithmic_windows` — sparse analysis scheduling.
- `MultiEpochsDataLoader` — performance-oriented DataLoader.

Also available via `perspic.calculator`: `CouplingCalculator`, `SamplewiseCalculator` (base class).

## Design notes

`analyzer()` sets `automatic_optimization = False` because its analysis metrics require extra forward/backward passes separate from the training pass: the `Linearizer` does its own `zero_grad -> forward -> backward` to get `||grad_L||^2`, and `SamplewiseCalculatorOpacus` runs `output_dim` extra forward+backward passes through the Opacus ghost-clipping hooks for `chi_net` (plus one extra forward for `chi_loss`). Manual optimization keeps these extra passes from corrupting Lightning's own gradient/optimizer state. If the wrapped module already uses manual optimization itself, `analyzer()` delegates to it instead of double-handling the optimizer step.

## Development

Install: `pip install -r requirements.txt -r dev-requirements.txt`

Test: `pytest`
- `tests/unit/` mirrors each `perspic` module 1:1 (`test_analyzer.py`, `test_linearizer.py`, `test_logger.py`, `test_samplewise.py`, `test_samplewise_functorch.py`, `test_samplewise_opacus.py`, `test_utils.py`), using `unittest.mock` and small synthetic Lightning modules.
- `tests/integration/` covers end-to-end/deployment scenarios: `test_analyzer_deployment.py`, `test_hutchinson_approximation.py`, `test_linearizer_deployment.py`, `test_lna_ntk_ground_truth.py`, `test_normalization_scaling.py`, `test_training_deployment.py`.

Lint/format: `pre-commit run --all-files` (runs `black`, `isort`, `flake8` in that order, `fail_fast: true`), or individually `black .`, `isort .`, `flake8`.

## Conventions

- Line length 88 (`black`, preview mode; `flake8` matches).
- `isort` uses the `black` profile.
- `flake8` ignores `W503, E203, E402, C901`; `examples/` is excluded from flake8 checks.
- Docstrings are predominantly Google-style (`Args:`, `Returns:`, `Raises:`).
- Type hints are used throughout function/method signatures.

## Examples

`examples/`:
- `cifar10.ipynb` — basic `analyzer()` usage: wrap a `LightningModule`, train on CIFAR-10, plot logged metrics.
- `cifar10_CrossReseponse.ipynb` — `cross_response=True` demo, measuring linear response against a held-out batch via a `CombinedLoader`.
- `batch_accumulation.ipynb` / `batch_accumulation_notes.md` — `micro_batch_size`/`effective_batch_size` gradient-accumulation demo and the design notes explaining the χ_loss aggregation fix and `effective_step`-based logging.
- `batch_size_scaling_analysis.ipynb` — synthetic sweep over batch size (repeated identical samples) verifying `chi_net`/`chi_loss` batch-size invariance directly against `SamplewiseCalculatorFunctorch`, independent of `analyzer()`.
- `measurement_batch_sweep.md` — design notes for `measure_dataloader`/`measure_batch_size`/`measure_subset_seed`, the independent (and sweepable) cross-response measurement batch size.
- `logging_scheduler.ipynb` — `logarithmic_windows` / `LogarithmicWindowSchedule` demo combined with LR scheduling.
- `mup_integration.ipynb` — integrating Maximal Update Parametrization (mup) with perspic.
- `core/hutchinson_convergence.py` — convergence of the Opacus Hutchinson trace-estimator approximation toward the exact per-sample gradient norm.
- `models/` — shared model zoo used by the notebooks above (`cnns.py`, `mlps.py`, `lightning_modules.py`, `utils.py`).
