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
        self.sample_calc = None
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
            # new self.model etc

        self.linearizer = Linearizer()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Compute samplewise metrics
        samples_results = self.sample_calc.compute(
            self.model,
            self.criterion,
            inputs,
            targets
        )

        # Use _backward_pass to handle different engines in the future
        self.manual_backward(loss)  # or loss.backward
        opt.step()

        # log block
        self.log("train_acc",
                 (outputs.argmax(dim=1) == targets).float().mean())
        self.log("train_loss", loss)
        self.log("per_sample_grad_norms",
                 samples_results["per_sample_grad_norms_network"])
        self.log("per_sample_grad_norms_loss",
                 samples_results["per_sample_grad_norms_loss"])

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
