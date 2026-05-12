# Perspic
A tool to study neural network training dynamics.

## What is Perspic?

Perspic provides efficient computation and logging of the loss-network-position (LNP) decomposition. Each of the components helps us to understand how neural networks learn. It integrates seamlessly with PyTorch Lightning by wrapping existing code, adding the analysis.

## Quick Start

```python
from perspic import analyzer
from pytorch_lightning import Trainer

# Wrap your Lightning module class
model = analyzer(
    lightning_module=YourLightningModule, # class
    model=your_model,
    sample_wise_engine="opacus"
)

trainer = Trainer(...)
trainer.fit(model, train_dataloader, val_dataloader)
```

## Installation

**Note:** Perspic is not yet published on PyPI. Install it locally with dependencies:

```bash
git clone https://github.com/zincware/perspic.git
cd perspic
python -m pip install .
```

## Learning & Examples

Start with [examples/README.md](examples/README.md) for comprehensive tutorials on core workflows, advanced features, and expert topics using CIFAR-10 and Vision Transformers.

## Documentation

- [Examples](examples/) — Jupyter notebooks demonstrating key features
- [Tests](tests/) — Unit and integration tests


## License

See [LICENSE](LICENSE)

