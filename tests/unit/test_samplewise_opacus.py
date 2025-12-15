import torch
import torch.nn as nn

from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus
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


class TestSamplewiseCalculatorOpacus:
    """Test Opacus calculator by cross-validating against functorch."""

    # --- Cross-validation tests (Opacus vs Functorch) ---

    def test_network_gradient_norms_match_functorch(self):
        """Verify Opacus matches functorch for network gradients (simple MLP)."""
        torch.manual_seed(42)
        batch_size = 8
        model = MLP(output_dim=2, sum_output=False, n_hidden=10)
        X = torch.randn(batch_size, 10)

        with BatchStatSnapshot(model, X):
            # Compare reduced (scalar) results
            opacus_reduced = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, X, reduce=True
                )
            )
            functorch_reduced = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                    model, X, reduce=True
                )
            )
            assert torch.allclose(
                opacus_reduced, functorch_reduced, atol=1e-4, rtol=1e-3
            )

            # Compare per-sample (unreduced) results
            opacus_per_sample = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, X, reduce=False
                )
            )
            functorch_per_sample = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                    model, X, reduce=False
                )
            )
            assert opacus_per_sample.shape == (batch_size,)
            assert functorch_per_sample.shape == (batch_size,)
            assert torch.allclose(
                opacus_per_sample, functorch_per_sample, atol=1e-4, rtol=1e-3
            )

    def test_loss_gradient_norms_match_functorch(self):
        """Verify Opacus matches functorch for loss gradients (simple MLP)."""
        torch.manual_seed(42)
        batch_size = 8
        model = MLP(output_dim=2, sum_output=False, n_hidden=10)
        X = torch.randn(batch_size, 10)
        y = torch.randint(0, 2, (batch_size,))
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        with BatchStatSnapshot(model, X):
            # Compare reduced (scalar) results
            opacus_reduced = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=True
                )
            )
            functorch_reduced = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=True
                )
            )
            assert torch.allclose(
                opacus_reduced, functorch_reduced, atol=1e-4, rtol=1e-3
            )

            # Compare per-sample (unreduced) results
            opacus_per_sample = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=False
                )
            )
            functorch_per_sample = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=False
                )
            )
            assert opacus_per_sample.shape == (batch_size,)
            assert functorch_per_sample.shape == (batch_size,)
            assert torch.allclose(
                opacus_per_sample, functorch_per_sample, atol=1e-4, rtol=1e-3
            )

    def test_batchnorm_mlp_matches_functorch(self):
        """Verify BatchNorm MLP handling matches functorch with BatchStatSnapshot."""
        torch.manual_seed(42)
        batch_size = 12
        input_dim = 10
        model = BatchNormMLP(input_dim=input_dim, hidden_dim=20, output_dim=2)
        X = torch.randn(batch_size, input_dim)
        y = torch.randint(0, 2, (batch_size,))
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        with BatchStatSnapshot(model, X):
            # Compare reduced (scalar) results
            opacus_network = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, X, reduce=True
                )
            )
            opacus_loss = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=True
                )
            )
            functorch_network = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                    model, X, reduce=True
                )
            )
            functorch_loss = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=True
                )
            )
            assert torch.allclose(
                opacus_network, functorch_network, atol=1e-4, rtol=1e-3
            )
            assert torch.allclose(opacus_loss, functorch_loss, atol=1e-4, rtol=1e-3)

            # Compare per-sample (unreduced) results
            opacus_network_ps = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, X, reduce=False
                )
            )
            opacus_loss_ps = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=False
                )
            )
            functorch_network_ps = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                    model, X, reduce=False
                )
            )
            functorch_loss_ps = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=False
                )
            )
            assert opacus_network_ps.shape == (batch_size,)
            assert opacus_loss_ps.shape == (batch_size,)
            assert functorch_network_ps.shape == (batch_size,)
            assert functorch_loss_ps.shape == (batch_size,)
            assert torch.allclose(
                opacus_network_ps, functorch_network_ps, atol=1e-4, rtol=1e-3
            )
            assert torch.allclose(
                opacus_loss_ps, functorch_loss_ps, atol=1e-4, rtol=1e-3
            )

    def test_cnn_model_matches_functorch(self):
        """Verify CNN with BatchNorm2d matches functorch with BatchStatSnapshot."""
        torch.manual_seed(42)
        batch_size = 7
        model = SimpleCNN()
        # Batch size 7, 3 channels, 32x32 image
        X = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        with BatchStatSnapshot(model, X):
            # Compare reduced (scalar) results
            opacus_network = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, X, reduce=True
                )
            )
            opacus_loss = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=True
                )
            )
            functorch_network = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                    model, X, reduce=True
                )
            )
            functorch_loss = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=True
                )
            )
            assert torch.allclose(
                opacus_network, functorch_network, atol=1e-4, rtol=1e-3
            )
            assert torch.allclose(opacus_loss, functorch_loss, atol=1e-4, rtol=1e-3)

            # Compare per-sample (unreduced) results
            opacus_network_ps = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, X, reduce=False
                )
            )
            opacus_loss_ps = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=False
                )
            )
            functorch_network_ps = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                    model, X, reduce=False
                )
            )
            functorch_loss_ps = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=False
                )
            )
            assert opacus_network_ps.shape == (batch_size,)
            assert opacus_loss_ps.shape == (batch_size,)
            assert functorch_network_ps.shape == (batch_size,)
            assert functorch_loss_ps.shape == (batch_size,)
            assert torch.allclose(
                opacus_network_ps, functorch_network_ps, atol=1e-4, rtol=1e-3
            )
            assert torch.allclose(
                opacus_loss_ps, functorch_loss_ps, atol=1e-4, rtol=1e-3
            )

    # --- Reduce parameter tests ---

    def test_reduce_parameter_network(self):
        """Test reduce=True vs reduce=False for network gradients."""
        torch.manual_seed(42)
        batch_size = 8
        model = MLP(output_dim=2, sum_output=False, n_hidden=10)
        X = torch.randn(batch_size, 10)

        with BatchStatSnapshot(model, X):
            # With reduce=True (default) -> scalar
            grad_norms_reduced = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, X, reduce=True
                )
            )
            assert grad_norms_reduced.ndim == 0, "reduce=True should return scalar"

            # With reduce=False -> per-sample norms
            grad_norms_per_sample = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, X, reduce=False
                )
            )
            assert grad_norms_per_sample.shape == (
                batch_size,
            ), "reduce=False should return (batch_size,)"

            # Sum of per-sample should equal reduced
            assert torch.allclose(grad_norms_reduced, grad_norms_per_sample.sum())

    def test_reduce_parameter_loss(self):
        """Test reduce=True vs reduce=False for loss gradients."""
        torch.manual_seed(42)
        batch_size = 8
        model = MLP(output_dim=2, sum_output=False, n_hidden=10)
        X = torch.randn(batch_size, 10)
        y = torch.randint(0, 2, (batch_size,))
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        with BatchStatSnapshot(model, X):
            # With reduce=True (default) -> scalar
            loss_grad_norms_reduced = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=True
                )
            )
            assert loss_grad_norms_reduced.ndim == 0, "reduce=True should return scalar"

            # With reduce=False -> per-sample norms
            loss_grad_norms_per_sample = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_loss(
                    model, loss_fn, X, y, reduce=False
                )
            )
            assert loss_grad_norms_per_sample.shape == (
                batch_size,
            ), "reduce=False should return (batch_size,)"

            # Sum of per-sample should equal reduced
            assert torch.allclose(
                loss_grad_norms_reduced, loss_grad_norms_per_sample.sum()
            )

    # --- Public API test ---

    def test_compute_public_api(self):
        """Test the public compute method returns expected structure."""
        torch.manual_seed(42)
        batch_size = 8
        model = MLP(output_dim=2, sum_output=False, n_hidden=10)
        X = torch.randn(batch_size, 10)
        y = torch.randint(0, 2, (batch_size,))
        loss_fn = nn.CrossEntropyLoss()

        calc = SamplewiseCalculatorOpacus()
        with BatchStatSnapshot(model, X):
            results = calc.compute(model, loss_fn, X, y)

        assert isinstance(results, dict)
        assert "batch_grad_norms_network" in results
        assert "batch_grad_norms_loss" in results
        assert results["batch_grad_norms_network"].ndim == 0
        assert results["batch_grad_norms_loss"].ndim == 0

    # --- In-place operation handling test ---

    def test_inplace_operations_handled_correctly(self):
        """Test that models with inplace=True operations work and state is preserved."""

        class InplaceModel(nn.Module):
            """Model with inplace=True ReLU operations."""

            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(inplace=True),
                    nn.Linear(20, 20),
                    nn.ReLU(inplace=True),
                    nn.Linear(20, 2),
                )

            def forward(self, x):
                return self.layers(x)

        torch.manual_seed(42)
        batch_size = 8
        model = InplaceModel()
        X = torch.randn(batch_size, 10)
        y = torch.randint(0, 2, (batch_size,))
        loss_fn = nn.CrossEntropyLoss()

        # Verify inplace=True before computation
        relu_modules = [m for m in model.modules() if isinstance(m, nn.ReLU)]
        assert len(relu_modules) == 2, "Model should have 2 ReLU modules"
        for relu in relu_modules:
            assert relu.inplace is True, "ReLU should have inplace=True before compute"

        # Run Opacus computation (this would crash without the fix)
        calc = SamplewiseCalculatorOpacus()
        with BatchStatSnapshot(model, X):
            results = calc.compute(model, loss_fn, X, y, normalize=False)

        # Verify inplace=True is restored after computation
        for relu in relu_modules:
            assert (
                relu.inplace is True
            ), "ReLU inplace should be restored to True after compute"

        # Verify we got valid results
        assert results["batch_grad_norms_network"].ndim == 0
        assert results["batch_grad_norms_loss"].ndim == 0
        assert results["batch_grad_norms_network"] > 0
        assert results["batch_grad_norms_loss"] > 0

        # Cross-validate with functorch (which handles inplace fine)
        torch.manual_seed(42)
        model_functorch = InplaceModel()

        with BatchStatSnapshot(model_functorch, X):
            functorch_network = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_network(
                    model_functorch, X, reduce=True
                )
            )
            functorch_loss = (
                SamplewiseCalculatorFunctorch._compute_per_sample_gradient_norm_loss(
                    model_functorch, loss_fn, X, y, reduce=True
                )
            )

        assert torch.allclose(
            results["batch_grad_norms_network"], functorch_network, atol=1e-4, rtol=1e-3
        ), "Opacus network gradient norms should match functorch for inplace model"
        assert torch.allclose(
            results["batch_grad_norms_loss"], functorch_loss, atol=1e-4, rtol=1e-3
        ), "Opacus loss gradient norms should match functorch for inplace model"
