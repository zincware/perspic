"""
PyTorch Lightning wrapper modules for training any PyTorch model.

This module provides flexible Lightning modules that can wrap any PyTorch nn.Module,
including the MLPs defined in MLPs.py, for easy training with PyTorch Lightning.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR
from torchmetrics.functional import accuracy


class ClassificationModule(pl.LightningModule):
    """
    A flexible Lightning module for classification tasks.

    Can wrap any PyTorch model and provides standard training/validation/test loops
    with configurable optimizers and schedulers.

    Args:
        model: Any PyTorch nn.Module to wrap
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        lr: Learning rate (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'sgd', 'adamw') (default: 'adam')
        scheduler_name: LR scheduler to use
            (None, 'onecycle', 'cosine', 'step') (default: None)
        weight_decay: Weight decay for optimizer (default: 0.0)
        momentum: Momentum for SGD optimizer (default: 0.9)
        scheduler_kwargs: Additional kwargs for the scheduler

    Example:
        >>> from models.MLPs import SimpleMLP
        >>> model = SimpleMLP()
        >>> lit_model = ClassificationModule(model, num_classes=10, lr=0.001)
        >>> trainer = pl.Trainer(max_epochs=10)
        >>> trainer.fit(lit_model, train_dataloader, val_dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 0.001,
        optimizer_name: str = "adam",
        scheduler_name: str = None,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        scheduler_kwargs: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.criterion = F.cross_entropy

        if scheduler_kwargs is None:
            self.scheduler_kwargs = {}
        else:
            self.scheduler_kwargs = scheduler_kwargs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.hparams.num_classes
        )

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.hparams.num_classes
        )

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.hparams.num_classes
        )

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Select optimizer
        if self.hparams.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_name}")

        # No scheduler
        if self.hparams.scheduler_name is None:
            return optimizer

        # Configure scheduler
        if self.hparams.scheduler_name.lower() == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.trainer.train_dataloader),
                **self.scheduler_kwargs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        elif self.hparams.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, **self.scheduler_kwargs
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.hparams.scheduler_name.lower() == "step":
            step_size = self.scheduler_kwargs.get("step_size", 10)
            gamma = self.scheduler_kwargs.get("gamma", 0.1)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler_name}")


class AdvancedClassificationModule(pl.LightningModule):
    """
    An advanced Lightning module with more features for classification.

    Includes:
    - Label smoothing
    - Mixup data augmentation (optional)
    - Gradient clipping
    - Custom logging

    Args:
        model: Any PyTorch nn.Module to wrap
        num_classes: Number of output classes
        lr: Learning rate
        optimizer_name: Optimizer to use
        scheduler_name: LR scheduler to use
        weight_decay: Weight decay
        momentum: Momentum for SGD
        label_smoothing: Label smoothing factor (default: 0.0)
        gradient_clip_val: Gradient clipping value (default: None)
        scheduler_kwargs: Additional kwargs for scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 0.001,
        optimizer_name: str = "adam",
        scheduler_name: str = None,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        label_smoothing: float = 0.0,
        gradient_clip_val: float = None,
        scheduler_kwargs: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        if scheduler_kwargs is None:
            self.scheduler_kwargs = {}
        else:
            self.scheduler_kwargs = scheduler_kwargs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Use cross entropy with label smoothing
        loss = F.cross_entropy(logits, y, label_smoothing=self.hparams.label_smoothing)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.hparams.num_classes
        )

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.hparams.num_classes
        )

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.hparams.num_classes
        )

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Select optimizer
        if self.hparams.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_name}")

        # No scheduler
        if self.hparams.scheduler_name is None:
            return optimizer

        # Configure scheduler
        if self.hparams.scheduler_name.lower() == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.trainer.train_dataloader),
                **self.scheduler_kwargs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        elif self.hparams.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, **self.scheduler_kwargs
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.hparams.scheduler_name.lower() == "step":
            step_size = self.scheduler_kwargs.get("step_size", 10)
            gamma = self.scheduler_kwargs.get("gamma", 0.1)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler_name}")

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        if self.hparams.gradient_clip_val is not None:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.hparams.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )


if __name__ == "__main__":
    # Example usage
    from perspic.models import DeepMLP, SimpleMLP

    print("=" * 60)
    print("Lightning Module Examples")
    print("=" * 60)

    # Example 1: Simple usage with Adam optimizer
    print("\n1. SimpleMLP with Adam optimizer:")
    model = SimpleMLP()
    lit_model = ClassificationModule(model, num_classes=10, lr=0.001)
    print(f"   Model: {lit_model.model.__class__.__name__}")
    print(f"   Optimizer: Adam")
    print(f"   Learning rate: {lit_model.hparams.lr}")

    # Example 2: DeepMLP with SGD and OneCycle scheduler
    print("\n2. DeepMLP with SGD + OneCycle scheduler:")
    model = DeepMLP()
    lit_model = ClassificationModule(
        model,
        num_classes=10,
        lr=0.05,
        optimizer_name="sgd",
        scheduler_name="onecycle",
        weight_decay=5e-4,
        momentum=0.9,
    )
    print(f"   Model: {lit_model.model.__class__.__name__}")
    print(f"   Optimizer: SGD (momentum={lit_model.hparams.momentum})")
    print(f"   Scheduler: OneCycle")
    print(f"   Weight decay: {lit_model.hparams.weight_decay}")

    # Example 3: Advanced module with label smoothing
    print("\n3. SimpleMLP with Advanced module + label smoothing:")
    model = SimpleMLP()
    lit_model = AdvancedClassificationModule(
        model,
        num_classes=10,
        lr=0.001,
        optimizer_name="adamw",
        scheduler_name="cosine",
        label_smoothing=0.1,
        gradient_clip_val=1.0,
    )
    print(f"   Model: {lit_model.model.__class__.__name__}")
    print(f"   Optimizer: AdamW")
    print(f"   Scheduler: Cosine Annealing")
    print(f"   Label smoothing: {lit_model.hparams.label_smoothing}")
    print(f"   Gradient clipping: {lit_model.hparams.gradient_clip_val}")

    print("\n" + "=" * 60)
    print("Ready to use with PyTorch Lightning Trainer!")
    print("=" * 60)
