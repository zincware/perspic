import pytest
import torch
import torch.nn as nn
import torch.utils.data as data
from nngeometry import GramMatrix

from perspic.calculator.samplewise import SamplewiseCalculatorFunctorch

torch.set_float32_matmul_precision("high")

# TODO: Test multiple model architectures


class MLP(nn.Module):
    """Simple 3 layer ReLU MLP for testing purposes."""

    def __init__(self, output_dim, sum_output=False, n_hidden=10):
        super().__init__()
        self.fc1 = nn.Linear(10, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, output_dim)
        self.sum_output = sum_output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        if self.sum_output:
            out = torch.sum(out, dim=1)
        return out


class TestTraceComputation:
    """Test MLP model and gradient computation."""

    @staticmethod
    def _compute_with_nngeometry(model, inputs, output_dim):
        """Compute trace using nngeometry."""
        batch_size = inputs.shape[0]
        loader = data.DataLoader(
            data.TensorDataset(inputs),
            batch_size=batch_size,
            shuffle=False,
        )
        K = GramMatrix(model=model, loader=loader)
        kernel = K.get_dense_tensor()
        matrix_kernel = kernel.reshape(batch_size * output_dim, batch_size * output_dim)
        trace = torch.trace(matrix_kernel)
        return trace

    @staticmethod
    def _compute_grad_norms(model, inputs):
        """Compute per-sample gradient norms."""
        calculator = SamplewiseCalculatorFunctorch()
        return calculator._compute_per_sample_gradient_norm_network(model, inputs)

    @staticmethod
    def _compute_loss_grad_norms(model, loss_fn, inputs, targets):
        """Compute per-sample gradient norms for loss."""
        calculator = SamplewiseCalculatorFunctorch()
        return calculator._compute_per_sample_gradient_norm_loss(
            model, loss_fn, inputs, targets
        )

    def test_forward(self):
        """Test the forward pass of the MLP model."""
        torch.manual_seed(42)
        model = MLP(output_dim=1, sum_output=False, n_hidden=10)
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 1)

    @pytest.mark.parametrize(
        "output_dim,sum_output,n_hidden,expected_output_dim",
        [
            (1, False, 10, 1),  # test_trace_computation_1
            (2, True, 10, 1),  # test_trace_computation_2
            (2, False, 10, 2),  # test_trace_computation_3
            (2, False, 100, 2),  # test_trace_computation_4
        ],
    )
    def test_trace_computation(
        self,
        output_dim,
        sum_output,
        n_hidden,
        expected_output_dim,
    ):
        """Test gradient norm computation matches nngeometry."""
        torch.manual_seed(42)
        model = MLP(
            output_dim=output_dim,
            sum_output=sum_output,
            n_hidden=n_hidden,
        )
        X = torch.randn(100, 10)

        grad_norms_network = self._compute_grad_norms(model, X)
        trace = self._compute_with_nngeometry(model, X, expected_output_dim)
        assert torch.allclose(grad_norms_network, trace)

    @pytest.mark.parametrize(
        "output_dim,sum_output,n_hidden",
        [
            (1, False, 10),  # test_loss_gradient_norm_computation_1
            (2, True, 10),  # test_loss_gradient_norm_computation_2
            (2, False, 10),  # test_loss_gradient_norm_computation_3
            (2, False, 100),  # test_loss_gradient_norm_computation_4
            (5, False, 50),  # test_loss_gradient_norm_computation_5
        ],
    )
    def test_loss_gradient_norm_computation(
        self,
        output_dim,
        sum_output,
        n_hidden,
    ):
        """Test loss gradient norm computation."""
        torch.manual_seed(42)
        model = MLP(
            output_dim=output_dim,
            sum_output=sum_output,
            n_hidden=n_hidden,
        )
        X = torch.randn(100, 10)
        # Create targets and loss matching the output shape
        if sum_output:
            # Scalar logit per sample -> use binary classification loss
            y = torch.randint(0, 2, (100,), dtype=torch.float32)
            loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        elif output_dim == 1:
            # Scalar logit per sample -> use regression loss
            y = torch.randn(100, 1)
            loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        else:
            # Multi-class logits with class dimension = output_dim
            y = torch.randint(0, output_dim, (100,))
            loss_fn = nn.CrossEntropyLoss(reduction="mean")

        # Compute loss gradient norms
        loss_grad_norms = self._compute_loss_grad_norms(model, loss_fn, X, y)
        # Basic sanity checks
        assert isinstance(loss_grad_norms, torch.Tensor)
        assert loss_grad_norms.ndim == 0  # Scalar
        assert loss_grad_norms.item() > 0  # Should be positive
