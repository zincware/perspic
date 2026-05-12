# Perspic Examples

Comprehensive examples demonstrating how to use perspic for analyzing neural network training dynamics.

---

## Quick Start

### Prerequisites

- PyTorch and PyTorch Lightning
- perspic (this package)
- CIFAR-10 dataset (auto-downloaded on first run)

### Running Your First Example

1. **Start with**: [cifar10.ipynb](cifar10.ipynb)
   - Full training pipeline with perspic analyzer
   - Shows how to wrap your model and collect metrics
   - ~30 minutes to run (or longer on CPU)

---

## Learning Path

Follow this order for best understanding:

### Level 1: Core Workflows

#### 1. [cifar10.ipynb](cifar10.ipynb)
**Duration**: ~30 min | **GPU**: Recommended (works on CPU but slow)

**What you'll learn**:
- How to wrap your PyTorch Lightning module with `analyzer()`
- What metrics perspic automatically computes (chi_net, chi_loss, coupling coefficient)
- How to configure CIFAR-10 datasets and dataloaders
- How to visualize training dynamics using perspic metrics

**Key Insight**: The `analyzer` wrapper is a drop-in enhancement for your LightningModule that requires minimal code changes.

---

### Level 2: Advanced Features

#### 2. [logging_scheduler.ipynb](logging_scheduler.ipynb)
**Duration**: ~20 min | **GPU**: Recommended

**Focus**: Efficient logging with logarithmic scheduling

**What you'll learn**:
- How to use `logarithmic_windows()` for sparse metric collection
- Why log-spaced sampling is efficient for long training runs
- How to configure `analysis_schedule` in the analyzer
- Trade-offs between metric frequency and computational cost

**Key Insight**: You don't need to compute metrics every step. Log-spaced schedules capture dynamics while reducing overhead.

---

#### 3. [cifar10_cross_response.ipynb](cifar10_cross_response.ipynb)
**Duration**: ~25 min | **GPU**: Recommended

**Focus**: Analyzing model response on fixed data

**What you'll learn**:
- How to set up a measurement set (fixed batch of test samples)
- How to use cross-response analysis to study sensitivity to perturbations
- Relationship between cross-response metrics and network learning
- `MultiEpochsDataLoader` for repeated cycling through measurement data

**Key Insight**: Cross-response analysis reveals how predictions respond to small weight changes, complementing training-time gradient analysis.

#### 4. [mup_integration.ipynb](mup_integration.ipynb)
**Duration**: ~40 min | **GPU**: Strongly recommended

**Focus**: Integration with Maximal Update Parametrization (μP)

**What you'll learn**:
- How μP enables hyperparameter transfer across model widths
- Why `CachedMuReadout` is needed for functorch compatibility
- How to use μP optimizers (`MuAdam`, `MuSGD`) with perspic
- How to use `set_base_shapes()` to mark width dimensions
- Applying this to WideResNet, ViT, and Llama-style transformers

**Key Insight**: Combine μP with perspic to verify that hyperparameters transfer correctly across model widths.

---

## Additional Examples

### [batch_size_scaling_analysis.ipynb](batch_size_scaling_analysis.ipynb)
**Duration**: ~5 min | **GPU**: Not required

**What it demonstrates**:
- How batch size affects gradient metrics
- Why repeated samples are useful for controlled experiments
- Expected scaling behavior: network gradients (linear) vs loss gradients (inverse)
- How perspic's normalization ensures batch-size invariant metrics

**Key Insight**: Using `reduction="mean"` in your loss combined with perspic's normalization produces batch-size invariant metrics.

This example is self-contained and doesn't require external data. Use it to understand batch-size scaling theory before moving to full training examples.

---

### [core/hutchinson_convergence.py](core/hutchinson_convergence.py)
**Duration**: ~2 min | **GPU**: Not required

**What it demonstrates**:
- Convergence of Hutchinson's trace estimator for per-sample gradient norms
- How random Rademacher projections approximate exact gradient norm computation
- Trade-off between estimation accuracy and computational cost
- Visualization of convergence rates and relative error

**Key Insight**: Hutchinson's estimator efficiently approximates squared gradient norms with just a handful of random projections, enabling fast per-sample gradient computations in perspic without looping over all output dimensions.

Run this script standalone to verify estimator accuracy: `python core/hutchinson_convergence.py`


---

## Tips & Troubleshooting

### "Why aren't my metrics being computed?"
- Check that `disable_analyzer=False` in the analyzer setup
- Verify that `analysis_schedule` includes the training steps you're running

### "The mup_integration notebook has a KeyboardInterrupt error"
- This is often a kernel timeout during torch import. Restart the kernel and re-run.

### "Can I modify the model architecture?"
- Yes! All CIFAR-10 notebooks focus on perspic mechanics, not specific architectures
- Swap models easily: try `WideResNet(28, 160)` or `BatchNormMLP(...)` instead of ViT

---

## Next Steps

- **Explore the source**: Look at `perspic/analyzer.py` to understand how metrics are computed
- **Run your own**: Use these notebooks as templates for your own models and datasets
- **Ask questions**: Open an issue if you have questions or suggestions

---

## File Structure

```
examples/
├── README.md                          # This file
├── __init__.py
├── models/                            # Example model definitions
│   ├── __init__.py
│   ├── cnns.py
│   ├── lightning_modules.py
│   ├── mlps.py
│   └── utils.py
├── batch_size_scaling_analysis.ipynb  # Theory + implementation
├── cifar10.ipynb                      # Core workflow
├── logging_scheduler.ipynb            # Efficient logging
├── cross10_cross_response.ipynb       # Response analysis
├── mup_integration.ipynb              # Advanced: μP + scaling laws
└── core/
    └── hutchinson_convergence.py      # Support utilities
```

---

## See Also

- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io)
- [Hutchinson Trace Estimation Paper](https://arxiv.org/abs/2312.14499)
- [Maximal Update Parametrization (μP) Paper](https://arxiv.org/abs/2203.03466)
