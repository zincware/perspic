import pytest
import torch
import torch.nn as nn
import torch.utils.data as data
from nngeometry import GramMatrix
from perspic.calculator.samplewise import SamplewiseCalculatorFunctorch


class MLP(nn.Module):
    """
    Simple 3 layer ReLU MLP for testing purposes.
    """

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


@pytest.fixture
def mlp_model_1():
    torch.manual_seed(42)
    model = MLP(output_dim=1, sum_output=False, n_hidden=10)
    return model


@pytest.fixture
def mlp_model_2():
    torch.manual_seed(42)
    model = MLP(output_dim=2, sum_output=False, n_hidden=10)
    return model


@pytest.fixture
def mlp_model_2_sum():
    torch.manual_seed(42)
    model = MLP(output_dim=2, sum_output=True, n_hidden=10)
    return model


@pytest.fixture
def mlp_model_hidden_100():
    torch.manual_seed(42)
    model = MLP(output_dim=2, sum_output=False, n_hidden=100)
    return model


def compute_per_sample_gradient_norm_network(model, inputs):
    calculator = SamplewiseCalculatorFunctorch()
    per_sample_grads = calculator._compute_per_sample_gradient_norm_network(
        model, inputs
    )
    return per_sample_grads


class TestMLP:
    def test_forward(self, mlp_model_1):
        """
        Test the forward pass of the MLP model.
        """
        x = torch.randn(5, 10)
        out = mlp_model_1(x)
        assert out.shape == (5, 1)

    @staticmethod
    def _compute_with_nngeometry(model, inputs, output_dim):
        batch_size = inputs.shape[0]
        loader = data.DataLoader(
            data.TensorDataset(inputs), batch_size=batch_size, shuffle=False
        )
        K = GramMatrix(model=model, loader=loader)
        kernel = K.get_dense_tensor()
        matrix_kernel = kernel.reshape(batch_size * output_dim,
                                       batch_size * output_dim)
        trace = torch.trace(matrix_kernel)
        return trace

    def test_trace_computation_1(self, mlp_model_1):
        X = torch.randn(100, 10)
        grad_norms_network = compute_per_sample_gradient_norm_network(
            mlp_model_1, X
        )
        trace = self._compute_with_nngeometry(
            mlp_model_1, X, 1
        )
        assert torch.allclose(grad_norms_network, trace)

    def test_trace_computation_2(self, mlp_model_2_sum):
        X = torch.randn(100, 10)
        grad_norms_network = compute_per_sample_gradient_norm_network(
            mlp_model_2_sum, X
        )
        trace = self._compute_with_nngeometry(
            mlp_model_2_sum, X, 1
        )
        assert torch.allclose(grad_norms_network, trace)

    def test_trace_computation_3(self, mlp_model_2):
        X = torch.randn(100, 10)
        grad_norms_network = compute_per_sample_gradient_norm_network(
            mlp_model_2, X
        )
        trace = self._compute_with_nngeometry(
            mlp_model_2, X, 2
        )
        assert torch.allclose(grad_norms_network, trace)

    def test_trace_computation_4(self, mlp_model_hidden_100):
        X = torch.randn(100, 10)
        grad_norms_network = compute_per_sample_gradient_norm_network(
            mlp_model_hidden_100, X
        )
        trace = self._compute_with_nngeometry(
            mlp_model_hidden_100, X, 2
        )
        assert torch.allclose(grad_norms_network, trace)
