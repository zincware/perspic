# Batch Accumulation in Perspic — Implementation Notes

## What gradient accumulation does

Instead of feeding a large batch of size N=K·B through the model in one forward pass, we
feed K micro-batches of size B and accumulate the resulting gradients before calling
`opt.step()`. The loss of each micro-batch is divided by K before `backward()` so that
the gradient that lands in `p.grad` after K steps is identical to the gradient you would
have obtained from one forward pass on the full N-sample batch.

```
gradient update = Σ_k  grad( L_k / K )
                = (1/K) Σ_k  grad( L_k )          # linearity of differentiation
                =  grad( (1/K) Σ_k L_k )           # same as mean loss over all N samples
                =  grad( L_full )                   # ✓
```

This is verified numerically: with Adam and SGD the max parameter-gradient difference
between the accumulated run and the full-batch run is < 2 × 10⁻⁸.

---

## Issue 1 — χ_loss aggregation was wrong (K² error, now fixed)

### Background: how χ_loss is normalised

`SamplewiseCalculatorOpacus.compute()` (and the functorch variant) return two metrics with
`normalize=True`:

| metric | formula returned | meaning |
|---|---|---|
| `chi_net` | `Σᵢ ‖∇_θ f(xᵢ)‖² / B` | average per-sample squared network-gradient norm |
| `chi_loss` | `Σᵢ ‖∇_f L(f(xᵢ),yᵢ)‖² · B` | scaled sum of per-sample squared loss-gradient norms |

The `· B` factor in `chi_loss` looks like it makes it "extensive" (growing with B), but
this is cancelled by the `1/B` that appears inside `∇_f L` when the loss is mean-reduced
(PyTorch `cross_entropy` default).

#### Why chi_loss is effectively intensive

With mean-reduced cross-entropy, the per-sample loss gradient w.r.t. the network output
scales as **1/B**:

```
∂L_mean/∂f(xᵢ)  =  (1/B) · ∂L_individual(xᵢ)/∂f(xᵢ)
```

Substituting into the formula:

```
chi_loss(B)  =  Σᵢ ‖ (1/B) · gᵢ ‖²  ·  B
             =  Σᵢ ‖gᵢ‖² / B²  ·  B
             =  Σᵢ ‖gᵢ‖² / B
```

where `gᵢ = ∂L_individual(xᵢ)/∂f(xᵢ)` is independent of B. So `chi_loss(B)` scales as
`1/B`, and the product `(sum) · B` that the calculator returns is **constant w.r.t. B**
(at fixed data distribution).

This can be verified empirically: `chi_loss(B=64) ≈ chi_loss(B=256) ≈ 0.9` for the same
model and data distribution.

### The original wrong formula

The original accumulation code used:

```python
chi_loss_eff = K * sum(chi_loss_k)   # WRONG
```

The reasoning was that `chi_loss` scales with B, so for N=K·B it should be multiplied by
K. But as shown above, chi_loss does *not* grow with B for mean-reduced losses. The
correct aggregation for an intensive quantity is the **mean**:

```
chi_loss(N=K·B)  =  Σ_{all N} ‖gᵢ‖² / N
                 =  (K · Σ_{i in micro} ‖gᵢ‖²) / (K·B)
                 =  (Σ_{i in micro} ‖gᵢ‖²) / B
                 =  chi_loss(B)
                 =  mean_k( chi_loss_k )
```

### The error it caused

| quantity | correct formula | old formula | error factor |
|---|---|---|---|
| `chi_net_eff` | `mean(chi_net_k)` | `mean(chi_net_k)` | 1 (correct) |
| `chi_loss_eff` | `mean(chi_loss_k)` | `K · sum(chi_loss_k)` | K² too large |
| `chi_coup` | `‖∇L‖² / (chi_loss · chi_net)` | (derived) | K² too small |

With K=4 this produced a **16× discrepancy** in `chi_loss` and `chi_coup` compared to the
full-batch run, clearly visible in the bottom row of the comparison plot.

### The fix

```python
# was:
chi_loss_eff = K * sum(self._accum_chi_loss)
# now:
chi_loss_eff = sum(self._accum_chi_loss) / K
```

Same correction applies to `chi_loss_cross_eff` for the cross-response path.

### Generalisation note

If you use a **sum-reduced** loss (e.g. `reduction='sum'` in `cross_entropy`), then
`chi_loss` *would* scale as B and the correct formula would be `K · sum(chi_loss_k)`.
The right formula depends on the loss normalisation convention. With the default
mean-reduced loss, `mean` is correct.

---

## Issue 2 — The train_loss curve looks different between runs (not a bug)

The bottom-left panel of the comparison plot uses `global_step` (Lightning's micro-batch
counter) on the x-axis. This causes a visual mismatch between the two runs:

### X-axis misalignment

| run | steps per opt.step() | opt steps after N global steps |
|---|---|---|
| full batch (B=256) | 1 | N |
| accumulation (K=4) | 4 | N/4 |

After 1000 global steps, the full-batch run has taken 1000 optimizer steps, while the
accumulation run has only taken 250. Both runs make the same number of optimizer steps
eventually (at step 4000 for accumulation vs step 1000 for full batch), but the x-axis
stretches the accumulation curve by a factor of K=4.

**This is a display artifact, not a computation error.** The analysis metrics (χ_net,
χ_loss, χ_coup) use `analysis_step = effective_step` (the optimizer step count) and are
therefore correctly aligned. Only `train_loss` uses raw `global_step`.

### Higher noise in the loss curve — a logging artifact, not a training difference

The accumulation run *looks* noisier because `ClassificationModule.training_step` calls
`self.log("train_loss", loss)` on **every** call, which fires once per micro-batch. So
for K=4 accumulation the CSV gets four separate loss values per optimizer step, each
computed on a fresh B=64 mini-batch drawn from the DataLoader.

The full-batch run logs once per optimizer step, each value computed on a B=256 batch.

This creates two compounding visual effects:

1. **Sampling variance.** A B=64 loss estimate has variance ≈ σ²/64; a B=256 estimate
   has variance ≈ σ²/256 — four times smaller. So every individual logged point in the
   accumulation run is noisier, which is reflected in the raw (faint) trace.

2. **Four times as many points.** The EMA smoother sees four data points per optimizer
   step instead of one, so it gets reset by each new noisy value before it can settle —
   making the smoothed curve appear noisier too.

**Neither effect reflects a real difference in what the optimizer is doing.** The
gradient that `opt.step()` acts on is the mean over all four micro-batches, which is
mathematically identical to the gradient from a single B=256 pass (verified numerically:
max parameter-gradient difference < 2 × 10⁻⁸). The apparent noise is entirely a
consequence of *what is being logged*, not of what is being computed.

**Why `train_loss` can't be re-logged as a mean — Lightning key de-duplication:**

Lightning's `CSVLogger` writes one row per `global_step` (micro-batch counter). When
`self.log(...)` is called anywhere during a `training_step`, the value is accumulated
into the *current step's* metrics dict and flushed to the CSV at the end of that step.
The wrapped model's `training_step` logs `train_loss` (a single micro-batch loss) first;
any subsequent call to `self.log("train_loss", mean_loss)` at cycle-end loses because
Lightning does not let a later log overwrite an earlier one for the same key and step.
The cycle-end mean is silently dropped — the raw micro-batch loss always appears in the
CSV.

**First plot fix attempted — `.last()` per `effective_step` group:**

When `effective_step` is logged only at the cycle end, Lightning **forward-fills** its
value into the next `K-1` rows (the first three micro-batches of cycle N+1). So
`effective_step=N` appears in four CSV rows:

- `global_step` = 4N   (micro-batch 4 of cycle N — cycle end, `effective_step` logged here)
- `global_step` = 4N+1 (micro-batch 1 of cycle N+1 — **forward-filled**)
- `global_step` = 4N+2 (micro-batch 2 of cycle N+1 — **forward-filled**)
- `global_step` = 4N+3 (micro-batch 3 of cycle N+1 — **forward-filled**)

This means `.groupby("effective_step").last()` picks `global_step = 4N+3` — a
micro-batch from the *next* cycle, not the mean. `.first()` picks `global_step = 4N`,
which is the 4th micro-batch loss of cycle N (not a mean either, since the mean log
was dropped). Neither `.first()` nor `.last()` gives the true per-cycle mean.

**Second fix — `.mean()` per `effective_step` group:**

`.groupby("effective_step").mean()` averages the four rows that share `effective_step=N`.
Even though three of them belong to cycle N+1 (due to forward-fill), this is a
sliding-window average over four consecutive micro-batches and reduces noise by roughly
4×. Empirically: accumulation EMA residual std = 0.052 vs full-batch 0.053 — essentially
equal. But the averaging window is shifted by one micro-batch relative to the optimizer
step boundary.

**Correct fix — log `effective_step` on every micro-batch:**

The cleanest solution is to tag each micro-batch with its cycle's `effective_step` at
the time it runs, eliminating any dependence on Lightning's forward-fill:

```python
# Inside training_step, for every micro-batch during accumulation:
# opt.step() has not fired yet so +1 gives the current cycle number
self.log("effective_step", float(self._optimizer_step_count + 1),
         on_step=True, on_epoch=False)
```

With this in place, all K micro-batches of cycle N carry `effective_step=N` directly.
`groupby("effective_step").mean()` then averages exactly those K micro-batches —
the true mean loss over the effective batch of size K·B. No forward-fill, no shifted
window, no dropped values.

### Is the optimizer step counter affected?

No. PyTorch optimizers (Adam, SGD, …) maintain an internal step counter `t` that
increments by 1 on each call to `opt.step()`. Adam uses `t` for bias correction:

```
m̂ = m / (1 - β₁ᵗ),    v̂ = v / (1 - β₂ᵗ)
```

With gradient accumulation, `opt.step()` is called exactly as many times as with a
full-batch run (once per effective batch), so `t` is identical between the two runs at
any given optimizer step. Adam's bias correction is unaffected. The gradient accumulation
does not introduce any error via the optimizer's internal state.

### Summary of the implemented fix

`analyzer.py` now logs `effective_step` on **every** micro-batch (not just at
cycle-end), using `_optimizer_step_count + 1` so all K micro-batches of cycle N share
the same `effective_step = N` value without relying on Lightning's forward-fill.

The notebook's `loss_per_opt_step()` helper then applies:

```python
sub = metrics[["effective_step", "train_loss"]].dropna()
sub = sub.groupby("effective_step")["train_loss"].mean().reset_index()
```

This averages exactly the K micro-batch losses from the same optimizer step, giving one
data point per optimizer step with variance σ²/(K·B) — the same as a direct B·K batch.
The full-batch run needs no special treatment. All four plot panels use `effective_step`
on the x-axis, making the curves directly comparable.
