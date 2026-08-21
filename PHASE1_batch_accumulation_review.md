# PHASE 1 — Batch-Accumulation Feature Audit

## Data-flow trace

Per micro-batch, [analyzer.py:262-365](perspic/analyzer.py#L262-L365) executes: `opt.zero_grad()` only when `_accumulation_count == 0` → analysis before-hook → wrapped `training_step` → `manual_backward(output / K)` → increment count → when `count >= K`: `opt.step()`, increment `_optimizer_step_count`, reset count, step per-step schedulers.

The analysis path ([_analyze_accumulated_step](perspic/analyzer.py#L503-L584)) decides `_analysis_active` on micro-batch 0 using `effective_step`, saves/restores `p.grad` clones around the grad-clobbering analysis passes, accumulates chi metrics and unscaled linearizer gradients per micro-batch, and finalizes on micro-batch K−1 — **before** the optimizer step, consistent with the single-step path, which also measures pre-step state. Exactly K entries accumulate per cycle; **no off-by-one in the accumulation or flush ordering itself**.

Math verified end-to-end:
- Gradient: `Σₖ ∇(Lₖ/K) = ∇(mean loss over N=K·B)` ✓ (assuming mean-reduced criterion, equal micro-batch sizes)
- `grad_norm_squared = ‖Σgₖ‖²/K² = ‖∇L_full‖²` ✓ — confirmed numerically by [test G](tests/unit/test_analyzer.py#L1248-L1307)
- `chi_net_eff = chi_loss_eff = mean over K` ✓ per the derivation in [batch_accumulation_notes.md](examples/batch_accumulation_notes.md). Note: **the old `K·sum` chi_loss bug (a K² error, 16× at K=4) is fixed only in the uncommitted working tree** — the committed branch tip still contains it. That fix needs to be committed.

## Verdict: ⚠️ Minor issues — core math correct; one label misalignment and several edge cases

## Correctness issues

1. **`effective_step` is off by one against everything it's compared with** — [analyzer.py:329-334](perspic/analyzer.py#L329-L334). Cycle N (0-indexed) tags its micro-batches `effective_step = N+1`, but the same cycle's analysis metrics log `analysis_step = N` ([analyzer.py:765](perspic/analyzer.py#L765)), and the full-batch comparison run's `train_loss` lands at Lightning `step = N`. The notebook plots loss by `effective_step` and chi by `analysis_step` on shared axes, so the accumulation loss curve is shifted +1 — visible at early steps on log axes.
   **Fix**: `self.log("effective_step", float(self._optimizer_step_count), ...)` — drop the `+1`.

2. **`_accum_step_losses` is dead code** — initialized at [analyzer.py:251](perspic/analyzer.py#L251), appended at [323](perspic/analyzer.py#L323), cleared at [345-346](perspic/analyzer.py#L345-L346) and [735](perspic/analyzer.py#L735), never read. Retains detached GPU scalars for nothing. **Fix**: delete all four sites.

3. **No guard against ragged micro-batches** — `output / K` assumes every micro-batch has exactly `micro_batch_size` samples. With `drop_last=False`, a smaller final batch makes the accumulated gradient a weighted (wrong) mean, and [analyzer.py:628](perspic/analyzer.py#L628) uses the *last* micro-batch's `x.shape[0]` for the logged `batch_size`/`effective_batch_size`. **Fix**: warn once in `training_step` when `batch[0].shape[0] != micro_batch_size`; document `drop_last=True`.

4. **Interrupted / partial cycles**: `_accumulation_count` isn't reset at epoch end, so cycles span epoch boundaries (self-consistent — gradients and analysis both span — but undocumented); a trailing partial cycle at end of training silently drops its gradients; `on_train_epoch_end` steps epoch-interval schedulers even mid-cycle. Additionally `_optimizer_step_count`/`_accumulation_count` are **not checkpointed**, so a resumed run resets `effective_step` to 0 and misaligns `analysis_schedule`. **Fix**: `on_save_checkpoint`/`on_load_checkpoint` for both counters; document epoch-spanning behavior.

5. **Question — mean-reduction assumption**: loss/K scaling *and* the mean-aggregation of `chi_loss` are only valid for a mean-reduced criterion (your notes acknowledge this). Nothing checks `criterion.reduction`; a sum-reduced criterion silently yields wrong gradients and a K²-wrong `chi_loss_eff`. Suggest a best-effort warning via `getattr(criterion, "reduction", "mean")`.

6. **Question**: `effective_step` is logged even when `log_metrics=False` — intentional (training metadata, not an analysis metric)? It's the only unconditional `self.log` in the class.

No race conditions: single-process Lightning, no threading/async anywhere in the accumulation flow.

## Efficiency issues

1. **Train-side linearizer passes are redundant** — [_accumulate_linearizer_grads](perspic/analyzer.py#L586-L623) adds one extra forward+backward per micro-batch, but at cycle end `p.grad` already holds `Σ∇(Lₖ)/K`, whose squared norm equals exactly the `‖Σgₖ‖²/K²` that [finalize](perspic/analyzer.py#L644-L647) computes. Reading `p.grad` just before `opt.step()` eliminates **K forward+backward passes per analysis cycle plus one full-model gradient buffer**. (Matches your existing hook-based-refactor design notes; measure side genuinely needs its own passes.) ~15% of analysis cost at 10 output dims; much more for low-output-dim models.
2. **`BatchStatSnapshot` pays a full dummy forward even with zero BatchNorm layers** — [utils.py:145-146](perspic/utils.py#L145-L146). For BN-free models: one wasted full-batch forward per analysis micro-batch. Early-out (but keep the `model.eval()` switch so dropout behavior is unchanged) — cheapest win in the feature.
3. **One GPU sync per parameter** — `.item()` inside per-parameter generator sums at [analyzer.py:644-647](perspic/analyzer.py#L644-L647), [675-682](perspic/analyzer.py#L675-L682), and [linearizer.py:78-79](perspic/calculator/linearizer.py#L78-L79), [103-107](perspic/calculator/linearizer.py#L103-L107). Sum on-device, call `.item()` once.
4. The loop itself is **O(K)** with in-place `acc.add_()` — no quadratic behavior, no unneeded copies beyond the required `saved_grads` clone (necessary because `sample_calc.compute` calls `model.zero_grad()`). Peak: ~3 concurrent full-gradient buffers (`saved_grads`, `_accum_grad_train`, `_accum_grad_measure`) — acceptable, worth a docstring note. `itertools`/`deque`/vectorization don't apply here; the buffers are parameter-shaped tensors handled correctly.

## Best-practice suggestions

- The train/measure branches of `_accumulate_linearizer_grads` are copy-paste duplicates; unify.
- Several tests drive `_before_training_step` by poking `_accumulation_count` directly — brittle coupling to private state; the `training_step`-driven style of `test_analysis_does_not_corrupt_training_gradients` is more robust.
- Typo "the the" in the manual-optimization warning ([analyzer.py:185](perspic/analyzer.py#L185)).
