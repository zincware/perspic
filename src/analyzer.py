import torch
from torch import nn
import pytorch_lightning as pl
from calculator.samplewise import (
    SamplewiseCalculatorFunctorch,
    SamplewiseCalculatorOpacus,
)
from calculator.linearizer import Linearizer


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
        optimizer=None,
        criterion=None,
        data_loader=None,
        sample_wise_engine: str = "functorch",
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_loader = data_loader
        self.sample_wise_calculator = None
        self.sample_wise_engine = sample_wise_engine

        # Turn off automatic optimization to handle optimizer steps manually
        self.automatic_optimization = False

        if sample_wise_engine not in ["opacus", "functorch"]:
            raise ValueError(
                "sample_wise_engine must be either 'opacus' or 'functorch'"
            )

        if sample_wise_engine == "functorch":
            self.samplewise_calculator = SamplewiseCalculatorFunctorch()
        elif sample_wise_engine == "opacus":
            self.samplewise_calculator = SamplewiseCalculatorOpacus()
            # get wrapped components with _get_wrappings and set them as 
            # new self.model etc

        self.linearizer = Linearizer()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # manual optimization logic
        # use _backward_pass to handle different engines
        return None

    def _backward_pass(self, loss):
        """Handle backward pass based on the samplewise engine."""
        backward_methods = {
            "opacus": lambda: loss.backward(),  # Opacus requires standard
            # backprop as it hooks into the autograd engine,
            # TODO: Check how to integrate with Lightings precision/strategy
            "functorch": lambda: self.manual_backward(loss),
        }

        method = backward_methods.get(self.samplewise_calculator.method)
        if method:
            method()
        else:
            # Default fallback
            loss.backward()

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        if self.data_loader is None:
            raise ValueError("Data loader is not provided.")
        return self.data_loader


class PerspicTrainer(pl.Trainer):
    def __init__(self, analyzer=None, *args, **kwargs):
        # Extract analyzer from kwargs before passing to parent
        self.analyzer = analyzer
        super().__init__(*args, **kwargs)
        # Custom initialization for PerspicTrainer can be added here

    def fit(self, model=None, **kwargs):
        # If no model is provided, use the stored analyzer
        if model is None:
            model = self.analyzer
        return super().fit(model, **kwargs)


if __name__ == "__main__":

    # minimal example sketch
    # model = ...
    # analyzer = Analyzer(model=model, sample_wise_engine="...")
    # trainer = PerspicTrainer(analyzer=analyzer, max_epochs=10)
    # trainer.fit()
    pass
