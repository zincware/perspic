# PHASE 2 — Full Project Review

**[SEVERITY: HIGH]** [samplewise_opacus.py:287-340](perspic/calculator/samplewise_opacus.py#L287-L340)
- **Issue**: `gs_model.remove_hooks()` and `_cleanup_opacus_leftovers()` are not in the `finally` block (only `_restore_inplace_ops` is).
- **Impact**: any exception mid-computation (CUDA OOM, the `NotImplementedError` for tied params) leaves ghost-clipping hooks attached — every subsequent training forward accumulates activations and per-sample state, silently corrupting training and growing memory.
- **Fix**:
```python
gs_model = None
try:
    ...
    gs_model = _GhostNormFastGradientClipping(...)
    ...
    return total_sq_norms.sum() if reduce else total_sq_norms
finally:
    if gs_model is not None:
        gs_model.remove_hooks()
    _cleanup_opacus_leftovers(model)
    _restore_inplace_ops(inplace_states)
```

**[SEVERITY: MEDIUM]** [samplewise_opacus.py:150-153](perspic/calculator/samplewise_opacus.py#L150-L153)
- **Issue**: `_cleanup_opacus_leftovers` deletes `param.norm_sample`, but opacus 1.5.4 (verified against installed source of `get_norm_sample`) stores `param._norm_sample`; the current check is dead.
- **Impact**: after the final output-dim iteration, `_norm_sample` tensors stay attached to every parameter indefinitely.
- **Fix**: add `if hasattr(param, "_norm_sample"): delattr(param, "_norm_sample")`.

**[SEVERITY: MEDIUM]** [utils.py:106-207](perspic/utils.py#L106-L207)
- **Issue**: `BatchStatSnapshot.__enter__` is not exception-safe — if the dummy forward raises, `__exit__` never runs.
- **Impact**: model left with `momentum=1.0`, BN layers in train mode, running stats clobbered, forward hooks still registered.
- **Fix**: wrap the body in `try/except`, restore saved state and remove hooks before re-raising (and add the no-BN early-out from Phase 1).

**[SEVERITY: MEDIUM]** [samplewise_opacus.py:366-367](perspic/calculator/samplewise_opacus.py#L366-L367)
- **Issue**: `_compute_per_sample_gradient_norm_loss` runs `model(inputs)` with grad enabled, then immediately detaches the output.
- **Impact**: builds and stores the full parameter autograd graph (all activations) that is never used — wasted memory every analysis step.
- **Fix**:
```python
with torch.no_grad():
    outputs = model(inputs)
outputs = outputs.requires_grad_(True)
```

**[SEVERITY: MEDIUM]** [coupling.py:33](perspic/calculator/coupling.py#L33)
- **Issue**: unguarded division `grad_norm_squared / (chi_loss * chi_net)`.
- **Impact**: zero gradients → `ZeroDivisionError` (floats) or silently logged `inf`/`nan` (tensors).
- **Fix**: return `float("nan")` when the denominator is 0.

**[SEVERITY: MEDIUM]** [logger.py:134-137](perspic/logger.py#L134-L137)
- **Issue**: every window whose tail exceeds `max_steps` is skipped — for `max_steps < base_window` even the step-0 window is dropped.
- **Impact**: `logarithmic_windows(max_steps=4)` returns an empty schedule; analysis silently never runs.
- **Fix**: warn when the resulting schedule is empty (or clamp the step-0 window). Related: overlapping windows keep their full step lists in `windows` even when `step_to_window` reassigns steps to a later window, so `window_width` overcounts.

**[SEVERITY: MEDIUM]** [analyzer.py:239-240](perspic/analyzer.py#L239-L240)
- **Issue**: accumulation/step counters not persisted across checkpoints (see Phase 1 #4).
- **Impact**: schedule misalignment and `zero_grad` boundary desync on resume.
- **Fix**: save/restore `_optimizer_step_count` and `_accumulation_count` in `on_save_checkpoint`/`on_load_checkpoint`.

**[SEVERITY: LOW]** — briefly:
- [analyzer.py:349-357](perspic/analyzer.py#L349-L357): scheduler stepping ignores `config.frequency`; relies on private `self._trainer`.
- [analyzer.py:315-317](perspic/analyzer.py#L315-L317): `output / K` breaks if the wrapped `training_step` returns Lightning's `{"loss": ...}` dict (pre-existing).
- [analyzer.py:415-501](perspic/analyzer.py#L415-L501): full analysis compute runs even when `log_metrics=False`, results discarded.
- [samplewise_opacus.py:178](perspic/calculator/samplewise_opacus.py#L178): Rademacher vectors hardcoded `float32` — mismatches float64/bf16 models.
- [samplewise_opacus.py:17-19](perspic/calculator/samplewise_opacus.py#L17-L19): import-time mutation of Opacus's global sampler registry affects any other Opacus user in the process.
- [mlps.py:103](examples/models/mlps.py#L103): mutable default argument `hidden_sizes=[1024, 512, 256]`.
- [lightning_modules.py:173](examples/models/lightning_modules.py#L173): `AdvancedClassificationModule` has no `criterion` attribute → incompatible with `analyzer()` (and its label smoothing wouldn't be visible to analysis regardless).
- [analyzer.py:17](perspic/analyzer.py#L17): `Optional[str]` on `sample_wise_engine` is misleading (`None` is rejected); the dynamically created `Analyzer` class also can't be unpickled / `load_from_checkpoint`-ed — worth a docstring note.
- Repo hygiene: `examples/cifar-10-python.tar.gz` (~170 MB) is untracked and **not** covered by the new `.gitignore` entries (only `cifar-10-batches-py/*` is); same for `examples/MNIST/` and `first_wrong_test_batch_accumulation.png` — one careless `git add .` away from a 170 MB commit.

## Top 5 prioritized actions

1. **Make the Opacus wrapper exception-safe** (`remove_hooks` + cleanup in `finally`) and fix the `_norm_sample` cleanup name — prevents silent training corruption after any analysis failure.
2. **Drop the `+1` from `effective_step`** and remove dead `_accum_step_losses` — restores x-axis alignment across runs and metrics. Also **commit the pending chi_loss `sum/K` fix**, which currently exists only in the working tree.
3. **Warn on ragged micro-batches and checkpoint the accumulation counters** — closes the remaining accumulation edge cases.
4. **`BatchStatSnapshot`: skip the dummy forward for BN-free models and make `__enter__` exception-safe** — cheapest meaningful speed + robustness win.
5. **Replace train-side `_accumulate_linearizer_grads` with a read of `p.grad` at cycle end** — K fewer forward+backward passes per analysis cycle and one fewer full-gradient buffer; first step of the hook-based refactor already sketched in your design notes.
