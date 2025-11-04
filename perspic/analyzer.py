import torch
from torch import nn
import pytorch_lightning as pl
from perspic.calculator.samplewise import (
    SamplewiseCalculatorFunctorch,
    SamplewiseCalculatorOpacus,
)
from perspic.calculator.linearizer import Linearizer
from typing import Optional
from lightning_fabric.utilities.types import _PATH


pl.seed_everything(42)


class Analyzer(pl.LightningModule):
    """
    model wrapper to monitor training behavior

    Args:
        model: model to be analyzed.
        optimizer: optimizer to be used for training.
        criterion: loss function to be used for training.
        data_loader: data loader for the training data.
        sample_wise_engine: engine to be used for sample-wise gradients. Either
            'opacus' or 'functorch'.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        data_loader,
        sample_wise_engine: str = "functorch",
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.sample_wise_engine = sample_wise_engine

        # Turn off automatic optimization to handle optimizer steps manually
        self.automatic_optimization = False

        if sample_wise_engine not in ["opacus", "functorch"]:
            raise ValueError(
                "sample_wise_engine must be either 'opacus' or 'functorch'"
            )

        if sample_wise_engine == "functorch":
            self.sample_calc = SamplewiseCalculatorFunctorch()
        elif sample_wise_engine == "opacus":
            self.sample_calc = SamplewiseCalculatorOpacus()
            # get wrapped components with _get_wrappings and set them as
            # new self.model etc.

        self.linearizer = Linearizer()

        # Set up optimizer
        self.optimizer = optimizer
        # Trainer check flag (must happen on first training step)
        self._check_trainer_on_first_step = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # One-time checks on first step (only gets called with fit())
        if self._check_trainer_on_first_step:
            self._first_step_checks()

        opt = self.optimizer
        opt.zero_grad()
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Compute samplewise metrics
        samples_results = self.sample_calc.compute(
            self.model, self.criterion, inputs, targets
        )

        probe_results = self.linearizer.probe_train_step(
            model=self.model,
            criterion=self.criterion,
            x=inputs,
            y=targets,
            eta=1e-5,
        )

        # Use standard backward when no trainer, manual_backward with trainer
        if self._trainer_attached:
            self.manual_backward(loss)
        else:
            loss.backward()

        opt.step()

        # log block
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("train_acc", acc)
        self.log("train_loss", loss)
        self.log(
            "batch_grad_norms_network",
            samples_results["batch_grad_norms_network"],
        )
        self.log(
            "batch_grad_norms_loss",
            samples_results["batch_grad_norms_loss"],
        )
        self.log("loss_value", probe_results[0])
        self.log("perturbed_loss_value", probe_results[1])
        self.log("actual_batch_size", inputs.shape[0])
        return None

    def _first_step_checks(self):
        """Perform one-time checks on the first training step."""
        # Check if trainer is attached
        trainer_obj = self.__dict__.get("trainer", None) or self.__dict__.get(
            "_trainer", None
        )
        self._trainer_attached = trainer_obj is not None
        self._check_trainer_on_first_step = False

    def _backward_pass(self, loss):
        """Handle backward pass based on the samplewise engine."""
        backward_methods = {
            "opacus": lambda: loss.backward(),  # Opacus requires standard
            # backprop as it hooks into the autograd engine,
            # TODO: Check how to integrate with Lightings precision/strategy
            "functorch": lambda: self.manual_backward(loss),
        }

        # Use the configured sample-wise engine string to select the method.
        engine_key = getattr(self, "sample_wise_engine", None)
        method = backward_methods.get(engine_key) if engine_key else None
        if method:
            method()
        else:
            # Default fallback: prefer trainer-aware manual backward if availb
            if getattr(self, "_trainer_attached", False):
                self.manual_backward(loss)
            else:
                loss.backward()

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        if self.data_loader is None:
            raise ValueError("Data loader is not provided.")
        return self.data_loader


class PerspicTrainer(pl.Trainer):
    def __init__(self, analyzer: "pl.LightningModule", **kwargs):
        super().__init__(**kwargs)
        self.analyzer = analyzer

    def fit(
        self,
        model=None,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=None,
        ckpt_path: Optional[_PATH] = None,
    ) -> None:
        """Override fit to use stored analyzer if no model provided."""
        model = self.analyzer

        # Validate model is not None before calling super
        if model is None:
            raise ValueError("no analyzer stored in trainer")

        super().fit(
            model,
            train_dataloaders,
            val_dataloaders,
            datamodule,
            ckpt_path,
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
    trainer = PerspicTrainer(analyzer=analyzer,
                             max_epochs=5,
                             log_every_n_steps=1)
    trainer.fit()
