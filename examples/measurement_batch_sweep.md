# Independent Measurement Batch Size — Implementation Notes

## The problem this solves

`analyzer(..., cross_response=True)` measures the model's linear response against a
held-out "measure" batch by expecting a dict batch `{"train": ..., "measure": ...}`,
produced with a `CombinedLoader({"train": train_loader, "measure": measure_loader},
mode="max_size_cycle")`. See `cifar10_CrossReseponse.ipynb`.

When gradient accumulation is active on the train side
(`micro_batch_size`/`effective_batch_size`, see `batch_accumulation_notes.md`), the
legacy cross-response code accumulates the measure gradient over the *same* K
micro-batches as the train side, and divides both by the same `K`
(`perspic/analyzer.py`, `_finalize_accumulated_analysis`). There was no way to:

1. Give the measure batch a size independent of the train effective batch size, or
2. Accumulate the measure side on its own when that size doesn't fit in one forward
   pass.

This was an open `TODO` in the code:

```python
# Accumulate linearizer gradients (measure side)
# TODO: How would a measure batchsize different to the effective batch size work here?
# !!! We would need to accumulate separately and then combine at the end.
```

This document describes the fix: `measure_dataloader` / `measure_batch_size` /
`measure_subset_seed`, a new, independent path that decouples the measurement batch
size from the training batch/accumulation entirely, and — as a second phase — lets one
analysis step sweep an **array** of measurement batch sizes.

The legacy `cross_response=True` + `CombinedLoader` path is untouched and still works
exactly as before. The new path is a separate, opt-in mechanism.

---

## Quick usage

### A single, independently sized measurement batch

```python
from torch.utils.data import DataLoader
from perspic import analyzer

# Sized however you like — independent of the train DataLoader's batch_size.
measure_loader = DataLoader(measurement_set, batch_size=500, drop_last=True)

model = analyzer(
    ClassificationModule,
    model=backbone,
    lr=0.1,
    measure_dataloader=measure_loader,
    measure_batch_size=2000,   # 4x measure-side accumulation (2000 // 500)
)
trainer.fit(model, train_dataloaders=train_loader)  # plain (x, y) loader, no CombinedLoader
```

`measure_batch_size` defaults to `measure_dataloader.batch_size` (no measure-side
accumulation) if omitted.

### Sweeping multiple measurement batch sizes

```python
from perspic import analyzer, logarithmic_windows

schedule = logarithmic_windows(max_steps=10_000, points_per_decade=5)

model = analyzer(
    ClassificationModule,
    model=backbone,
    lr=0.1,
    measure_dataloader=measure_loader,
    measure_batch_size=[500, 1000, 2000, 4000],
    measure_subset_seed=0,          # reproducible subset draws across runs
    analysis_schedule=schedule,     # see the warning below
)
```

Each analyzed step now logs one `cross_*` metric set **per swept size**, suffixed
`@bs{S}`: `cross_chi_net@bs500`, `cross_chi_net@bs1000`, ...,
`cross_grad_dot_product@bs4000`, etc.

**Warning:** sweeping without a logarithmic `analysis_schedule` runs the *entire*
sweep at *every* analyzed step (every step, if `analyze_every`/`analysis_schedule` are
both unset), which is expensive — each swept size does its own forward+backward passes
per micro-batch chunk. `analyzer()` emits a `UserWarning` at construction time if you
set a `measure_batch_size` list without also setting `analysis_schedule`.

Sizes below `measure_dataloader.batch_size` are valid too — they're measured as a
single direct pass instead of being rejected:

```python
measure_loader = DataLoader(measurement_set, batch_size=128, drop_last=True)

model = analyzer(
    ClassificationModule,
    model=backbone,
    lr=0.1,
    measure_dataloader=measure_loader,
    measure_batch_size=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
    measure_subset_seed=0,
    analysis_schedule=schedule,
)
# 4..128 -> single direct pass each (K_measure=1)
# 256, 512, 1024 -> accumulated (K_measure=2, 4, 8)
```

---

## How it works

### 1. An independent, persistent measure data source

`measure_dataloader` is any `DataLoader`. Unlike the train batch, the analyzer does
**not** expect it bundled into the training batch via `CombinedLoader`. Instead the
`Analyzer` owns its own iterator over `measure_dataloader`
(`_next_measure_micro_batch`), lazily created on first use and transparently refilled
(`iter(...)` again) whenever it's exhausted — so a finite measure dataset just cycles.
A `MultiEpochsDataLoader` (see `perspic/utils.py`) works too and is recommended for
small measurement sets, since it avoids re-spawning DataLoader workers every cycle.

Because the measure source is independent, the training batch is a **plain `(x, y)`
tuple** when `measure_dataloader` is set — no `CombinedLoader` dict, no `"measure"`
key. (The two mechanisms are mutually exclusive per run: pick `cross_response=True` +
`CombinedLoader`, *or* `measure_dataloader`.)

The measure **micro-batch size** — the largest chunk pulled from the loader in one
forward/backward pass — is inferred from `measure_dataloader.batch_size`. Use
`drop_last=True` (or a dataset size divisible by `batch_size`) so every pulled
micro-batch is full; a short final batch raises a `ValueError` rather than silently
building an undersized pool.

### 2. Sizing and validation

`measure_dataloader.batch_size` is the **maximum single-pass batch size** — the
largest batch your GPU can process in one forward+backward pass. `measure_batch_size`
accepts an `int` (single size) or a `list[int]` (sweep), and each entry `S` is handled
by one of two regimes depending on how it compares to that max single-pass size
(`micro = measure_dataloader.batch_size`):

- **`S <= micro`** — measured with a **single direct pass** on exactly `S` samples,
  no accumulation (this deliberately runs the GPU below its max capacity; there's
  nothing to validate here beyond `S` being a positive integer).
- **`S > micro`** — `S` must be an **exact multiple** of `micro`; measured by
  accumulating `S // micro` passes of size `micro` each into one combined measurement
  (the same rule as `effective_batch_size` vs `micro_batch_size` on the train side).

```
chunk_size = min(S, micro)
K_measure  = S // chunk_size
```

This single formula covers both regimes: `chunk_size = S` (so `K_measure = 1`, a lone
direct pass) when `S <= micro`, and `chunk_size = micro` (so `K_measure = S // micro`,
accumulated passes) when `S > micro`. `K_measure` is computed independently per swept
size and is completely decoupled from the train side's `accumulation_steps`. A run can
combine train-side accumulation (`micro_batch_size`/`effective_batch_size`) with
measure-side accumulation (`measure_dataloader`/`measure_batch_size`) freely — the two
`K`s do not need to match, and in general won't. A sweep like
`measure_batch_size=[4, 8, 16, 32, 64, 128, 256, 512, 1024]` with a 128-sample max
single pass measures `4, ..., 128` as direct passes (`K_measure=1` each) and
`256, 512, 1024` as accumulated measurements (`K_measure=2, 4, 8` respectively) — all
in the same analyzed step, each producing exactly one measurement.

### 3. Gather once, then subset — largest to smallest

Naively, sweeping N sizes could mean N independent pulls from the measure loader per
analyzed step (S₁ samples for size 1, S₂ for size 2, ...) — wasteful, and it makes
smaller sizes' samples *unrelated* to larger sizes' samples, which is usually not what
you want when studying how a metric depends on batch size.

Instead, on every analyzed step, `_measure_response` gathers **one pool** of
`S_max = max(measure_batch_size)` samples from the persistent iterator — pulling
`ceil(S_max / micro)` micro-batches (at least one, even when `S_max < micro`, i.e. the
whole sweep is below the max single-pass size) and slicing the concatenation down to
exactly `S_max` samples — and processes every requested size **largest → smallest**:

- The largest size uses the whole pool.
- Every smaller size `S` uses a **seed-fixable random subset** of `S` samples drawn
  from that *same* pool, via a single `torch.Generator` created once at `analyzer()`
  construction time and seeded (optionally) by `measure_subset_seed`. The generator is
  *not* reseeded between steps or between sizes, so a full run's sequence of subset
  draws is reproducible end-to-end given the same seed — rerun the same training script
  with the same `measure_subset_seed` and every swept metric matches.

Processing largest-first (rather than, say, smallest-first or in list order) is what
makes the smaller sizes' samples an actual *subset* of the larger sizes' samples,
rather than an independently-drawn batch — useful when you want to see how a metric
changes as you add more samples to the same pool, not how it varies across unrelated
draws.

```
pool = pull(ceil(S_max / micro)) micro-batches, concatenated, sliced to S_max samples
for S in sorted(measure_batch_size, reverse=True):
    if S == S_max:
        subset = pool                                          # use it all
    else:
        subset = pool[ randperm(S_max, generator=measure_gen)[:S] ]
    chunk_size = min(S, micro)
    K_measure  = S // chunk_size                                # 1 if S <= micro
    for chunk in chunks(subset, chunk_size):                    # K_measure chunks
        accumulate gradient + per-sample chi over `chunk`
    combine and log cross_* metrics (suffixed @bs{S} if sweeping)
```

Each chunk is wrapped in its own `BatchStatSnapshot(self.model, chunk)`
(`perspic/utils.py`), so BatchNorm statistics are frozen to *that chunk's own*
statistics before computing its per-sample gradients. This is a deliberate difference
from the legacy `cross_response=True` path, which freezes the measure computation to
the *train* batch's statistics — the independent path has no train batch to borrow
statistics from once the measure size diverges from the train size, so each measure
chunk uses its own.

### 4. Combining the metrics

The aggregation semantics mirror the existing **single-step** (no accumulation)
reference case — not the legacy accumulated-cross-response path's per-micro-batch
geometric-mean-then-average, which conflates the train and measure micro-batch
granularities. For each measure size `S`:

```
chi_net_measure(S)  = mean over the S's K_measure chunks of batch_grad_norms_network
chi_loss_measure(S) = mean over the S's K_measure chunks of batch_grad_norms_loss

cross_chi_net(S)  = sqrt( chi_net_self(train)  * chi_net_measure(S)  )   # compute_cross_metrics
cross_chi_loss(S) = sqrt( chi_loss_self(train) * chi_loss_measure(S) )

grad_train_mean   = (train accumulated gradient) / K_train     # K_train = 1 without accumulation
grad_measure_mean(S) = (measure accumulated gradient for S) / K_measure

cross_grad_dot_product(S) = <grad_train_mean, grad_measure_mean(S)>
cross_loss(S)              = mean loss over the S measure samples
cross_chi_coup(S)          = cross_grad_dot_product(S) / (cross_chi_loss(S) * cross_chi_net(S))
```

Same intensive/extensive reasoning as train-side accumulation applies here (see
"Issue 1" in `batch_accumulation_notes.md`): both `chi_net` and `chi_loss` are
computed with `normalize=True`, which makes the *correct* aggregate over accumulated
chunks the **mean**, not `K_measure * sum(...)`.

In the train-side-accumulated case, `_measure_response` is called from
`_finalize_accumulated_analysis` with `grad_train_mean = accumulated_train_grad /
accumulation_steps` (the same accumulated train gradient already used for the train
`chi_coup`). In the non-accumulated (single-step) case, the train gradient normally
gets discarded inside `Linearizer.compute()`, so `_analyze_single_step` recomputes it
once explicitly (a single extra forward/backward, only on analyzed steps) before
calling `_measure_response`.

`_measure_response` always saves and restores `self.model`'s gradients around its own
forward/backward passes (mirroring the existing analysis save/restore pattern in
`_analyze_accumulated_step`), so it never corrupts the live (possibly partially
accumulated) training gradient — this is covered by
`test_measure_backward_does_not_corrupt_training_grad` in `tests/unit/test_analyzer.py`.

### 5. Logging keys

| case | keys logged |
|---|---|
| single `measure_batch_size` (int, or a 1-element list) | `cross_chi_net`, `cross_chi_loss`, `cross_chi_coup`, `cross_loss`, `cross_grad_dot_product`, `cross_batch_size` — identical to the legacy `cross_response=True` keys |
| sweep (`measure_batch_size` list, length > 1) | the same set, suffixed `@bs{S}` per size, e.g. `cross_chi_net@bs500`, `cross_chi_net@bs2000`, ... |

`cross_effective_batch_size` is intentionally **not** logged for the independent
measure path: `batch_size` in the logged metric is already the full measure size `S` —
multiplying by the *train* `accumulation_steps` (what the generic logging helper does
for the train side) would be meaningless here, since `K_measure` is independent of
`K_train`.

---

## Validation rules (raised at `analyzer()` construction time)

- `measure_dataloader.batch_size` must not be `None` (i.e. it must use automatic
  batching).
- Every value in `measure_batch_size` must be a positive integer. Values `<=
  measure_dataloader.batch_size` need no further constraint (single direct pass).
  Values `>` it must be an exact multiple of it (measured via accumulation).
- `measure_batch_size` / `measure_subset_seed` may only be set together with
  `measure_dataloader`.
- A `measure_batch_size` list of length > 1 without `analysis_schedule` triggers a
  `UserWarning` (see the sweep-cost warning above).

## Edge cases

- **Partial final measure batch.** If `measure_dataloader` has `drop_last=False` and
  its dataset size isn't divisible by `batch_size`, the last batch of an epoch is
  short. Pulling it into the pool would silently build a short/misaligned pool, so
  `_next_measure_micro_batch` raises a `ValueError` instead, recommending
  `drop_last=True`.
- **Measure dataset smaller than `S_max`.** The persistent iterator just cycles, so the
  pool may contain repeated samples. This is fine for most uses but worth knowing if
  you're sizing `measure_batch_size` close to (or larger than) the measurement
  dataset's size.
- **Every swept size below the max single-pass size (`S_max < micro`).** The pool
  gather still pulls at least one micro-batch (`ceil(S_max / micro) = 1`) and slices
  it down to `S_max` samples — it never pulls zero micro-batches.
- **`disable_analyzer=True`.** `_measure_response` is never called (the whole analysis
  hook is skipped), so the measure loader is never touched — zero overhead.
- **`log_metrics=False`.** `_measure_response` is gated on `log_metrics` (matching the
  "no consumer, skip the expensive sweep" intent), so no measure-side computation runs
  at all in that case.

## Where to look in the code

- `perspic/analyzer.py`: `__init__` validation block ("Independent measure data
  source"), `_next_measure_micro_batch`, `_measure_response`, and the two call
  sites in `_analyze_single_step` and `_finalize_accumulated_analysis`.
- `tests/unit/test_analyzer.py`: `TestIndependentMeasureResponse` — validation,
  single-measure logging, the `K_measure`-vs-`K_train` mean divisor, a numeric
  gradient-dot-product reference test, subset-seed determinism, sweep suffixing, and
  iterator-cycling/partial-batch edge cases.
- `tests/integration/test_analyzer_deployment.py`: `TestAnalyzerWithIndependentMeasure`
  — end-to-end training with a real `Trainer`, combined with train-side accumulation,
  a sweep under a logarithmic schedule, and cross-run reproducibility.
