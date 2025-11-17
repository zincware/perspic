from typing import Optional

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import _PATH
from torch import nn

from perspic.calculator.linearizer import Linearizer
from perspic.calculator.samplewise import (SamplewiseCalculatorFunctorch,
                                           SamplewiseCalculatorOpacus)

pl.seed_everything(42)


class Analyzer(pl.LightningModule):
    """
    A wrapper around a PyTorch Lightning model to add analysis capabilities.
    This class wraps an existing LightningModule and adds functionality to compute
    samplewise gradients and linearizer probes to compute Collective Variables (CVs)
    during training.

    Args:
        model (pl.LightningModule): The original LightningModule to be wrapped.
        sample_wise_engine (str): The engine to use for samplewise gradient computation.
            Options are "functorch" or "opacus". Default is "functorch".
    """

    def __init__(
        self, model: pl.LightningModule, sample_wise_engine: str = "functorch"
    ):
        super().__init__()

        # Store the original model
        self.model = model
        
        # Store the samplewise engine choice
        self.sample_wise_engine = sample_wise_engine

        # Check if the model's training_step uses manual optimization
        if not model.automatic_optimization:
            raise Warning(
                "The wrapped model uses manual optimization. "
                "Gradient Updates will be delegated to the the wrapped model's training_step."
            )
            self.delegate_optimization = True
        else:
            self.delegate_optimization = False
        self.automatic_optimization = False  # We handle optimization manually

        # Initialize calculators
        if sample_wise_engine == "functorch":
            self.sample_calc = SamplewiseCalculatorFunctorch()
        elif sample_wise_engine == "opacus":
            self.sample_calc = SamplewiseCalculatorOpacus()
        else:
            raise ValueError(
                "sample_wise_engine must be either 'opacus' or 'functorch'"
            )

        self.linearizer = Linearizer()

        # Copy hyperparameters from original model
        if hasattr(model, "hparams"):
            self.save_hyperparameters(dict(model.hparams))

    def forward(self, x):
        """Delegate forward to the wrapped model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Wrapped training step with monitoring."""
        opt = self.optimizers()
        opt.zero_grad()

        # --- BEFORE: Compute samplewise metrics and linearizer probe ---
        x, y = batch

        # Compute samplewise metrics
        samples_results = self.sample_calc.compute(
            self.model, 
            self.model.criterion if hasattr(self.model, "criterion") else None,
            x,
            y,
        )

        # Linearizer probe
        probe_results = self.linearizer.probe_train_step(
            model=self.model,
            criterion=(
                self.model.criterion if hasattr(self.model, "criterion") else None
            ),
            x=x,
            y=y,
            eta=1e-5,
        )

        # --- CORE: Original training step ---
        original_loss = self.model.training_step(batch, batch_idx)

        if not self.delegate_optimization:
            # Backward pass
            self.manual_backward(original_loss)
            # Optimizer step
            opt.step()

        # --- AFTER: Log computed metrics ---
        self.log(
            "batch_grad_norms_network", samples_results["batch_grad_norms_network"]
        )
        self.log("batch_grad_norms_loss", samples_results["batch_grad_norms_loss"])
        self.log("loss_value", probe_results[0])
        self.log("perturbed_loss_value", probe_results[1])

        return original_loss

    def validation_step(self, batch, batch_idx):
        """Delegate validation to wrapped model."""
        return self.model.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """Delegate test to wrapped model."""
        return self.model.test_step(batch, batch_idx)

    def configure_optimizers(self):
        """Delegate optimizer configuration to wrapped model."""
        return self.model.configure_optimizers()
    
    def __getattr__(self, name):
        """Fallback: delegate any other methods to wrapped model."""
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


if __name__ == "__main__":
    # Example usage
    model = nn.Linear(10, 2)
    x, y = torch.randn(32, 10), torch.randint(0, 2, (32,))
    from torch.utils.data import DataLoader, TensorDataset

    data_loader = DataLoader(TensorDataset(x, y), batch_size=8)

    analyzer = Analyzer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        data_loader=data_loader,
        sample_wise_engine="functorch",
    )
    trainer = PerspicTrainer(analyzer=analyzer, max_epochs=5, log_every_n_steps=1)
    trainer.fit()
