import torch
import torch.nn as nn
import torch.nn.functional as F

from perspic.calculator.linearizer import (
    ApproximateLinearizer,
    ExactLinearizer,
    Linearizer,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleModel()
criterion = nn.CrossEntropyLoss()

x = torch.randn(4, 10)
y = torch.randint(0, 5, (4,))

# =============================================================================
# ApproximateLinearizer (alias: Linearizer)
# Uses virtual gradient steps to approximate the linear response
# =============================================================================
print("=" * 60)
print("ApproximateLinearizer (virtual step method)")
print("=" * 60)

eta_array = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
approx_linearizer = ApproximateLinearizer(eta_array=eta_array)

print(f"exact_linearizer: {approx_linearizer.exact_linearizer}")

approx_results = approx_linearizer.compute(
    model=model,
    criterion=criterion,
    x=x,
    y=y,
)

print("\nResults: {eta: (loss, perturbed_loss, delta_loss)}")
for eta, (loss, perturbed_loss, delta_loss) in approx_results.items():
    print(
        f"  eta={eta:.0e}: loss={loss:.6f}, perturbed={perturbed_loss:.6f}, delta={delta_loss:.6e}"
    )

# Verify Linearizer is an alias for ApproximateLinearizer
assert (
    Linearizer is ApproximateLinearizer
), "Linearizer should be an alias for ApproximateLinearizer"

# =============================================================================
# ExactLinearizer
# Computes ||∇L||² directly (cheaper and exact)
# =============================================================================
print("\n" + "=" * 60)
print("ExactLinearizer (gradient norm method)")
print("=" * 60)

exact_linearizer = ExactLinearizer()

print(f"exact_linearizer: {exact_linearizer.exact_linearizer}")

exact_results = exact_linearizer.compute(
    model=model,
    criterion=criterion,
    x=x,
    y=y,
)

print("\nResults: {eta: (loss, perturbed_loss, delta_loss)}")
loss, perturbed_loss, delta_loss = exact_results[-1]
print(
    f"  eta=-1 (exact): loss={loss:.6f}, perturbed={perturbed_loss:.6f}, delta={delta_loss:.6e}"
)
print(f"  ||∇L||² = {-delta_loss:.6f}")

# =============================================================================
# Comparison: ApproximateLinearizer should converge to ExactLinearizer
# =============================================================================
print("\n" + "=" * 60)
print("Comparison: delta_loss / eta should converge to -||∇L||²")
print("=" * 60)

grad_norm_squared = -exact_results[-1][2]
print(f"Exact ||∇L||² = {grad_norm_squared:.6f}")
print("\nApproximate delta_loss / eta:")
for eta, (_, _, delta_loss) in approx_results.items():
    ratio = delta_loss / eta
    error = abs(ratio - (-grad_norm_squared)) / grad_norm_squared * 100
    print(f"  eta={eta:.0e}: delta/eta={ratio:.6f}, error={error:.2f}%")
