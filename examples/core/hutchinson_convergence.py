"""Demonstrate convergence of Hutchinson's trace estimator for gradient norms.

This script shows that the approximate computation of per-sample gradient norms
using random Rademacher projections converges to the exact computation (looping
over all output dimensions) as the number of projections increases.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus


class SimpleMLP(nn.Module):
    """A simple MLP for demonstration."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    input_dim = 32
    output_dim = 10

    # Create model and inputs
    model = SimpleMLP(input_dim=input_dim, output_dim=output_dim).to(device)
    model.eval()  # Set to eval mode for consistent behavior

    inputs = torch.randn(batch_size, input_dim, device=device)

    # Compute exact per-sample gradient norms (full loop over output dimensions)
    exact_norms = SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
        model, inputs, reduce=False, approximate_with_n=None
    )
    exact_total = exact_norms.sum().item()

    print(f"Exact total squared gradient norm: {exact_total:.6f}")
    print()

    # Test different numbers of random projections
    n_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    n_repeats = 10  # Average over multiple runs for stability

    approx_means = []
    approx_stds = []

    for n in n_values:
        results = []
        for _ in range(n_repeats):
            approx_norms = (
                SamplewiseCalculatorOpacus._compute_per_sample_gradient_norm_network(
                    model, inputs, reduce=False, approximate_with_n=n
                )
            )
            # Hutchinson gives an estimate of the total norm directly
            approx_total = approx_norms.sum().item()
            results.append(approx_total)

        mean_val = sum(results) / len(results)
        std_val = (sum((r - mean_val) ** 2 for r in results) / len(results)) ** 0.5

        approx_means.append(mean_val)
        approx_stds.append(std_val)

        rel_error = abs(mean_val - exact_total) / exact_total * 100
        print(
            f"n={n:4d}: approx={mean_val:.6f} ± {std_val:.6f}, "
            f"rel. error={rel_error:.2f}%"
        )

    # Create convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Approximate vs Exact values
    ax1 = axes[0]
    ax1.axhline(y=exact_total, color="r", linestyle="--", linewidth=2, label="Exact")
    ax1.errorbar(
        n_values,
        approx_means,
        yerr=approx_stds,
        fmt="o-",
        capsize=5,
        label="Approximate (mean ± std)",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Random Projections (n)")
    ax1.set_ylabel("Total Squared Gradient Norm")
    ax1.set_title("Convergence of Hutchinson's Estimator")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Relative error
    ax2 = axes[1]
    rel_errors = [abs(m - exact_total) / exact_total * 100 for m in approx_means]
    ax2.plot(n_values, rel_errors, "o-", color="green", linewidth=2, markersize=8)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Random Projections (n)")
    ax2.set_ylabel("Relative Error (%)")
    ax2.set_title("Relative Error vs Number of Projections")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./hutchinson_convergence.png", dpi=150)
    plt.show()

    print()
    print("Plot saved to examples/hutchinson_convergence.png")


if __name__ == "__main__":
    main()
