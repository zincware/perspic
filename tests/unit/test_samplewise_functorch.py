import pytest
import torch
import torch.nn as nn
import torch.utils.data as data
from nngeometry import GramMatrix

from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.utils import BatchStatSnapshot

torch.set_float32_matmul_precision("high")


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


class BatchNormMLP(nn.Module):
    """MLP with BatchNorm for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleCNN(nn.Module):
    """Simple CNN with BatchNorm for testing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 10),
        )

    def forward(self, x):
        return self.layers(x)


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

    @staticmethod
    def _compute_loss_grad_norms_naive(model, loss_fn, inputs, targets):
        """Compute per-sample gradient norms for loss w.r.t. OUTPUTS using a naive loop."""
        grads = []

        for i in range(inputs.shape[0]):
            # Handle batch dimension for single sample
            x_i = inputs[i].unsqueeze(0)
            # Handle targets shape
            if targets.ndim > 1:
                y_i = targets[i].unsqueeze(0)
            else:
                y_i = targets[i].unsqueeze(0)

            # We need gradients w.r.t. output, not parameters.
            # Run forward pass (no need for parameter gradients)
            with torch.no_grad():
                out = model(x_i)

            # Enable gradient tracking for output to compute dL/dout
            out.requires_grad_(True)
            loss = loss_fn(out, y_i)

            # Compute gradients w.r.t. output
            grad_out = torch.autograd.grad(loss, out)[0]

            # Compute squared norm
            norm_sq = (grad_out.flatten() ** 2).sum()
            grads.append(norm_sq)

        return torch.stack(grads).sum()

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
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        elif output_dim == 1:
            # Scalar logit per sample -> use regression loss
            y = torch.randn(100, 1)
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            # Multi-class logits with class dimension = output_dim
            y = torch.randint(0, output_dim, (100,))
            loss_fn = nn.CrossEntropyLoss(reduction="sum")

        # Compute loss gradient norms
        loss_grad_norms = self._compute_loss_grad_norms(model, loss_fn, X, y)

        # Compute with naive loop
        loss_grad_norms_naive = self._compute_loss_grad_norms_naive(
            model, loss_fn, X, y
        )

        # Basic sanity checks
        assert isinstance(loss_grad_norms, torch.Tensor)
        assert loss_grad_norms.ndim == 0  # Scalar
        assert loss_grad_norms.item() > 0  # Should be positive

        # Check against naive implementation
        assert torch.allclose(loss_grad_norms, loss_grad_norms_naive, atol=1e-5)

    def test_batchnorm_model(self):
        """Test gradient computation with BatchNorm layers."""
        torch.manual_seed(42)
        model = BatchNormMLP(input_dim=10, hidden_dim=20, output_dim=2)
        X = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        # Without BatchStatSnapshot, this would fail or be incorrect for per-sample gradients
        # because BatchNorm couples samples in train mode.
        # We use the snapshot to fix stats and decouple samples.
        with BatchStatSnapshot(model, X):
            grad_norms = self._compute_grad_norms(model, X)
            loss_grad_norms = self._compute_loss_grad_norms(model, loss_fn, X, y)

            # Verify against naive loop
            # Note: Naive loop needs to run in eval mode with fixed stats to match
            # what BatchStatSnapshot does (which puts model in eval mode with fixed stats)
            # However, BatchStatSnapshot modifies the model in-place to have those stats.
            # So we can just run the naive loop on the model *inside* the context.
            loss_grad_norms_naive = self._compute_loss_grad_norms_naive(
                model, loss_fn, X, y
            )

        assert torch.allclose(loss_grad_norms, loss_grad_norms_naive, atol=1e-5)
        assert grad_norms.item() > 0

    def test_cnn_model(self):
        """Test gradient computation with CNN and BatchNorm."""
        torch.manual_seed(42)
        model = SimpleCNN()
        # Batch size 5, 3 channels, 32x32 image
        X = torch.randn(5, 3, 32, 32)
        y = torch.randint(0, 10, (5,))
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        with BatchStatSnapshot(model, X):
            grad_norms = self._compute_grad_norms(model, X)
            loss_grad_norms = self._compute_loss_grad_norms(model, loss_fn, X, y)

            loss_grad_norms_naive = self._compute_loss_grad_norms_naive(
                model, loss_fn, X, y
            )

        assert torch.allclose(loss_grad_norms, loss_grad_norms_naive, atol=1e-4)
        assert grad_norms.item() > 0

    def test_compute_public_api(self):
        """Test the public compute method."""
        torch.manual_seed(42)
        model = MLP(output_dim=2, sum_output=False, n_hidden=10)
        X = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        loss_fn = nn.CrossEntropyLoss()

        calculator = SamplewiseCalculatorFunctorch()
        results = calculator.compute(model, loss_fn, X, y)

        assert isinstance(results, dict)
        assert "batch_grad_norms_network" in results
        assert "batch_grad_norms_loss" in results
        assert results["batch_grad_norms_network"].ndim == 0
        assert results["batch_grad_norms_loss"].ndim == 0

    def test_reduce_parameter_network(self):
        """Test that reduce parameter works correctly for network gradients."""
        torch.manual_seed(42)
        model = MLP(output_dim=2, sum_output=False, n_hidden=10)
        X = torch.randn(10, 10)

        # With reduce=True (default) -> scalar
        grad_norms_reduced = self._compute_grad_norms(model, X)
        assert grad_norms_reduced.ndim == 0, "reduce=True should return scalar"

        # With reduce=False -> per-sample norms
        calculator = SamplewiseCalculatorFunctorch()
        grad_norms_per_sample = calculator._compute_per_sample_gradient_norm_network(
            model, X, reduce=False
        )
        assert grad_norms_per_sample.shape == (
            10,
        ), "reduce=False should return (batch_size,)"

        # Sum of per-sample should equal reduced
        assert torch.allclose(grad_norms_reduced, grad_norms_per_sample.sum())

    def test_reduce_parameter_loss(self):
        """Test that reduce parameter works correctly for loss gradients."""
        torch.manual_seed(42)
        model = MLP(output_dim=2, sum_output=False, n_hidden=10)
        X = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        # With reduce=True (default) -> scalar
        loss_grad_norms_reduced = self._compute_loss_grad_norms(model, loss_fn, X, y)
        assert loss_grad_norms_reduced.ndim == 0, "reduce=True should return scalar"

        # With reduce=False -> per-sample norms
        calculator = SamplewiseCalculatorFunctorch()
        loss_grad_norms_per_sample = calculator._compute_per_sample_gradient_norm_loss(
            model, loss_fn, X, y, reduce=False
        )
        assert loss_grad_norms_per_sample.shape == (
            10,
        ), "reduce=False should return (batch_size,)"

        # Sum of per-sample should equal reduced
        assert torch.allclose(loss_grad_norms_reduced, loss_grad_norms_per_sample.sum())


class Toy3DModel(nn.Module):
    """MLP whose output is reshaped to (batch, seq_len, vocab_size) -- see
    the identically-named model in test_samplewise_opacus.py."""

    def __init__(self, input_dim=10, n_hidden=10, seq_len=4, vocab_size=6):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, seq_len * vocab_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out.reshape(x.shape[0], self.seq_len, self.vocab_size)


class IndependentPositionModel(nn.Module):
    """Embedding -> Linear applied per-position: output position t depends
    ONLY on token t's own id, with no mixing across positions -- see the
    identically-named model in test_samplewise_opacus.py for why this is
    what makes the padded-vs-unpadded equality test below valid."""

    def __init__(self, vocab_size: int, embed_dim: int, n_classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.head(self.embed(x))


class TestIgnoreIndexMasking:
    """Tests for perspic's new support for `-100`-masked targets (WS5a) on
    the functorch backend -- mirrors test_samplewise_opacus.py's
    TestIgnoreIndexMasking. Functorch has no approximate mode (it always
    computes the full Jacobian via jacrev), so there is only one "exact
    mode" style equality to check.
    """

    def test_constructor_default_ignore_index(self):
        calc = SamplewiseCalculatorFunctorch()
        assert calc.ignore_index == -100

    def test_constructor_accepts_ignore_index_override(self):
        calc = SamplewiseCalculatorFunctorch(ignore_index=None)
        assert calc.ignore_index is None

    def test_unmasked_batch_compute_matches_pre_feature_formula(self):
        """No -100 anywhere in targets -> compute() must reproduce the
        pre-feature formula (n_elements = targets.numel(), no masking
        anywhere) exactly -- functorch has no randomness anywhere in this
        path, so any deviation would mean the new masking logic changed
        behavior for callers who never asked for it."""
        torch.manual_seed(5)
        model = Toy3DModel(input_dim=6, n_hidden=6, seq_len=3, vocab_size=4)
        X = torch.randn(5, 6)
        y = torch.randint(0, 4, (5, 3))  # no -100 present

        def loss_fn(outputs, targets):
            return nn.functional.cross_entropy(
                outputs.reshape(-1, outputs.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )

        pre_feature_net = (
            SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                model, X, reduce=True
            )
        )
        pre_feature_loss = (
            SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_loss(
                model, loss_fn, X, y, reduce=True
            )
        )
        n_elements_old = y.numel()
        expected_net = pre_feature_net / n_elements_old
        expected_loss = pre_feature_loss * n_elements_old

        calc = SamplewiseCalculatorFunctorch()  # default ignore_index=-100
        result = calc.compute(model, loss_fn, X, y, normalize=True)

        assert torch.equal(result["batch_grad_norms_network"], expected_net)
        assert torch.equal(result["batch_grad_norms_loss"], expected_loss)

    def test_padded_masked_matches_unpadded_per_sample(self):
        """V=5, T=4: a padded-and-masked batch's per-sample chi_net values
        must equal the values computed by running each sample's own
        (shorter, unpadded) sequence through the model individually."""
        torch.manual_seed(11)
        vocab_size, embed_dim, n_classes, seq_len = 5, 4, 5, 4
        model = IndependentPositionModel(vocab_size, embed_dim, n_classes)

        real_lens = [2, 3]
        x_padded = torch.randint(0, vocab_size, (2, seq_len))
        y_padded = torch.full((2, seq_len), -100, dtype=torch.long)
        for i, rl in enumerate(real_lens):
            y_padded[i, :rl] = torch.randint(0, n_classes, (rl,))

        mask, n_elements = SamplewiseCalculatorFunctorch.resolve_target_mask(
            y_padded, ignore_index=-100
        )
        assert mask is not None
        assert n_elements.item() == sum(real_lens)

        padded_per_sample = (
            SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                model, x_padded, reduce=False, mask=mask
            )
        )

        unpadded_per_sample = torch.stack(
            [
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                    model, x_padded[i, :rl].unsqueeze(0), reduce=True
                )
                for i, rl in enumerate(real_lens)
            ]
        )

        assert torch.allclose(
            padded_per_sample, unpadded_per_sample, atol=1e-5, rtol=1e-4
        )
