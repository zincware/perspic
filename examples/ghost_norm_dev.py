import copy

import torch
import torch.nn as nn
from opacus.grad_sample import (
    GradSampleModule,
    register_grad_sampler,
    register_norm_sampler,
)
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.utils.module_utils import requires_grad, trainable_parameters
from opacus.utils.per_sample_gradients_utils import (
    check_per_sample_gradients_are_correct,
)

from perspic.calculator.samplewise_functorch import SamplewiseCalculatorFunctorch
from perspic.utils import BatchStatSnapshot


# Register the sampler
@register_grad_sampler(nn.BatchNorm1d)
@register_grad_sampler(nn.BatchNorm2d)
@register_grad_sampler(nn.BatchNorm3d)
def compute_batch_norm_grad_sample(
    layer: nn.modules.batchnorm._BatchNorm,
    activations: list[torch.Tensor],
    backprops: torch.Tensor,
) -> dict[nn.Parameter, torch.Tensor]:
    """
    Computes per-sample gradients for BatchNorm layers when running in eval mode
    (frozen statistics).
    """
    activations = activations[0]
    # 1. Get the fixed statistics
    mean = layer.running_mean
    var = layer.running_var
    eps = layer.eps

    # 2. Reshape mean/var to match activation dimensions for broadcasting
    # activations shape: [N, C, D1, D2, ...]
    # mean/var shape: [C] -> [1, C, 1, 1, ...]
    view_shape = [1, layer.num_features] + [1] * (activations.dim() - 2)
    mean = mean.view(view_shape)
    var = var.view(view_shape)

    # 3. Normalize activations: x_hat = (x - mu) / sqrt(var + eps)
    x_hat = (activations - mean) / torch.sqrt(var + eps)

    # 4. Compute gradients
    # For bias (beta): sum backprops over all spatial dims
    # For weight (gamma): sum (backprops * x_hat) over all spatial dims

    # We want to sum over all dimensions except 0 (batch) and 1 (channel)
    sum_dims = list(range(2, activations.dim()))

    if len(sum_dims) > 0:
        grad_weight = torch.sum(backprops * x_hat, dim=sum_dims)
        grad_bias = torch.sum(backprops, dim=sum_dims)
    else:
        # For 1D input (N, C) without spatial dims (rare for BN but possible)
        grad_weight = backprops * x_hat
        grad_bias = backprops

    ret = {}
    if layer.weight is not None:
        ret[layer.weight] = grad_weight
    if layer.bias is not None:
        ret[layer.bias] = grad_bias

    return ret


@register_norm_sampler(nn.BatchNorm1d)
@register_norm_sampler(nn.BatchNorm2d)
@register_norm_sampler(nn.BatchNorm3d)
def compute_batch_norm_norm_sample(
    layer: nn.modules.batchnorm._BatchNorm,
    activations: list[torch.Tensor],
    backprops: torch.Tensor,
) -> dict[nn.Parameter, torch.Tensor]:

    # Reuse the gradient computation because for BN,
    # computing the grad and taking the norm is as efficient as it gets.
    # (Unlike Linear where we can avoid computing the outer product)

    grads = compute_batch_norm_grad_sample(layer, activations, backprops)

    ret = {}
    for param, grad in grads.items():
        # grad shape: [N, C]
        # norm shape: [N]
        ret[param] = grad.norm(2, dim=1)

    return ret


class GhostNormGradSampleModule(GradSampleModule):
    """
    Extension of GradSampleModule that allows computing per-sample gradients
    even when the model is in eval mode.

    This is necessary when using BatchStatSnapshot, which freezes batch statistics
    by switching the model to eval mode. Standard Opacus skips gradient sampling
    in eval mode.
    """

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: list[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if (
            not requires_grad(module)
            # or not module.training  <-- Removed this check to allow eval mode
            or not torch.is_grad_enabled()
        ):
            return

        if not self.hooks_enabled:
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in forward_input])

        for _, p in trainable_parameters(module):
            p._forward_counter += 1


class GhostNormFastGradientClipping(GradSampleModuleFastGradientClipping):
    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: list[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if (
            not requires_grad(module)
            # or not module.training  <-- Removed
            or not torch.is_grad_enabled()
            or not self.hooks_enabled
        ):
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in forward_input])

        for _, p in trainable_parameters(module):
            p._forward_counter += 1
            if (
                self.use_ghost_clipping
                and p._forward_counter > 1
                and type(module) in self.NORM_SAMPLERS
            ):
                raise NotImplementedError(
                    "Parameter tying is not supported with Ghost Clipping"
                )

    def get_squared_norm_sample(self) -> torch.Tensor:
        """
        Returns the squared L2 norm of the per-sample gradients.
        ||grad L(x_i)||^2
        """
        return self.get_norm_sample() ** 2


def compute_microbatch_gradients(model, x):
    microbatch_grads = {}
    criterion = nn.L1Loss(reduction="mean")

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.microbatch_grad = []

    # Iterate over samples
    for i in range(x.shape[0]):
        x_i = x[i : i + 1]  # Keep batch dim
        model.zero_grad()
        out = model(x_i)
        loss = criterion(out, torch.zeros_like(out))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.microbatch_grad.append(param.grad.detach().clone())

    # Stack
    for name, param in model.named_parameters():
        if param.requires_grad:
            microbatch_grads[name] = torch.stack(param.microbatch_grad)

    return microbatch_grads


def test_batch_norm_ghost_norm():
    # Test with BatchNorm
    bn_model = nn.Sequential(
        nn.Linear(10, 20), nn.BatchNorm1d(20), nn.ReLU(), nn.Linear(20, 1)
    )
    x_bn = torch.randn(32, 10)

    print("Testing BatchNorm gradient sampling...")

    # 1. Use BatchStatSnapshot to fix statistics to the current batch
    with BatchStatSnapshot(bn_model, x_bn):
        # Clone for microbatch BEFORE wrapping to avoid copying hooks
        import copy

        mb_model = copy.deepcopy(bn_model)

        # 2. Wrap with GhostNormGradSampleModule (strict=False to allow BatchNorm)
        # We use our custom class to allow gradient sampling in eval mode
        gs_bn_model = GhostNormGradSampleModule(bn_model, strict=False)

        # 3. Compute Opacus gradients
        criterion = nn.L1Loss(reduction="mean")
        gs_bn_model.zero_grad()
        out = gs_bn_model(x_bn)
        loss = criterion(out, torch.zeros_like(out))
        loss.backward()

        # 4. Compute Microbatch gradients (using the clone)
        mb_grads = compute_microbatch_gradients(mb_model, x_bn)

        print("\nVerifying gradients...")
        all_correct = True
        for name, param in gs_bn_model.named_parameters():
            if (
                param.requires_grad
                and hasattr(param, "grad_sample")
                and param.grad_sample is not None
            ):
                # Strip _module. prefix if present
                clean_name = name.replace("_module.", "")

                if clean_name in mb_grads:
                    opacus_grad = param.grad_sample
                    mb_grad = mb_grads[clean_name]

                    if not torch.allclose(opacus_grad, mb_grad, atol=1e-5, rtol=1e-4):
                        print(f"Gradient mismatch for {name} ({clean_name})")
                        print(f"Max diff: {(opacus_grad - mb_grad).abs().max()}")
                        all_correct = False
                    else:
                        print(f"Gradient match for {name} ({clean_name})")
                else:
                    print(
                        f"Parameter {name} ({clean_name}) not found in microbatch grads"
                    )

        if all_correct:
            print("\nSUCCESS: All per-sample gradients match micro-batch gradients.")
        else:
            print("\nFAILURE: Some gradients do not match.")

    print("\nTesting GhostNormFastGradientClipping (Norm Computation)...")

    # Create a fresh model for the second test to avoid hook conflicts
    bn_model_fast = nn.Sequential(
        nn.Linear(10, 20), nn.BatchNorm1d(20), nn.ReLU(), nn.Linear(20, 1)
    )
    # Copy weights from original model to ensure we can compare with mb_grads
    bn_model_fast.load_state_dict(bn_model.state_dict())

    # 1. Use BatchStatSnapshot to fix statistics to the current batch
    with BatchStatSnapshot(bn_model_fast, x_bn):
        # 2. Wrap with GhostNormFastGradientClipping
        gs_bn_model_fast = GhostNormFastGradientClipping(bn_model_fast, strict=False)

        # 3. Compute Opacus norms
        criterion = nn.L1Loss(reduction="mean")
        gs_bn_model_fast.zero_grad()
        out = gs_bn_model_fast(x_bn)
        loss = criterion(out, torch.zeros_like(out))
        loss.backward()

        # 4. Verify norms against microbatch gradients
        # We can reuse mb_grads from previous step since model and input are same

    print("\nVerifying norms...")
    all_correct_norms = True

    # Verify total squared norm
    print("Checking total squared norm...")
    opacus_sq_norms = gs_bn_model_fast.get_squared_norm_sample()

    mb_sq_norms = torch.zeros_like(opacus_sq_norms)
    for name, mb_grad in mb_grads.items():
        mb_sq_norms += mb_grad.reshape(mb_grad.shape[0], -1).pow(2).sum(dim=1)

    if not torch.allclose(opacus_sq_norms, mb_sq_norms, atol=1e-4, rtol=1e-3):
        print(f"Total squared norm mismatch")
        print(f"Max diff: {(opacus_sq_norms - mb_sq_norms).abs().max()}")
        all_correct_norms = False
    else:
        print(f"Total squared norm match")

    if all_correct_norms:
        print("\nSUCCESS: All per-sample gradient norms match.")
    else:
        print("\nFAILURE: Some norms do not match.")

    # Compute the norms instead for the loss, for the network outputs

    class SingleOutputModel(nn.Module):
        def __init__(self, base_model, dim):
            super().__init__()
            self.base_model = base_model
            self.dim = dim

        def forward(self, x):
            out = self.base_model(x)
            return out[:, self.dim]  # Keep batch dim

    # Create a fresh model for the second test to avoid hook conflicts
    bn_model_fast = nn.Sequential(
        nn.Linear(10, 20), nn.BatchNorm1d(20), nn.ReLU(), nn.Linear(20, 2)
    )
    # Copy weights from original model to ensure we can compare with mb_grads
    with BatchStatSnapshot(bn_model_fast, x_bn):
        mb_model = copy.deepcopy(bn_model_fast)

    with BatchStatSnapshot(bn_model_fast, x_bn):
        output_dim = bn_model_fast[-1].out_features
        mb_model = copy.deepcopy(bn_model_fast)  # Move inside context

        total_opacus_sq_norms = torch.zeros(x_bn.shape[0])

        for dim in range(output_dim):
            # Create fresh model copy to avoid hook accumulation
            model_copy = copy.deepcopy(bn_model_fast)
            single_output_model = SingleOutputModel(model_copy, dim)
            # Use loss_reduction='sum' since we're doing out.sum().backward()
            gs_model = GhostNormFastGradientClipping(
                single_output_model, strict=False, loss_reduction="sum"
            )

            gs_model.zero_grad()
            out = gs_model(x_bn)
            out.sum().backward()  # This gives per-sample grads via opacus

            total_opacus_sq_norms += gs_model.get_squared_norm_sample()

        # Use functorch-based SamplewiseCalculatorFunctorch for ground truth
        # This computes ||∇_θ f(x_i)||² directly using jacrev + vmap
        # Note: The public method sums over samples, so we need per-sample values
        # Replicate the internal computation to get per-sample norms
        inputs_unsqueezed = x_bn.unsqueeze(1)  # Due to vmap
        per_sample_grads = (
            SamplewiseCalculatorFunctorch._compute_per_sample_gradient_network_sum(
                mb_model, inputs_unsqueezed
            )
        )
        # Compute per-sample gradient magnitude (L2 norm) - without final sum
        per_sample_grad_magnitudes = torch.stack(
            [(g**2).sum(dim=tuple(range(1, g.ndim))) for g in per_sample_grads.values()]
        ).sum(
            dim=0
        )  # Sum across parameters, shape: (batch_size,)
        functorch_sq_norms = per_sample_grad_magnitudes

        # Final comparison
        print(f"\nComparing network gradient norms:")
        print(f"Total Opacus (loop over dims): {total_opacus_sq_norms}")
        print(f"Functorch (direct Jacobian):   {functorch_sq_norms}")
        print(
            f"Ratio (Opacus / Functorch):    {total_opacus_sq_norms / functorch_sq_norms}"
        )

        if torch.allclose(
            total_opacus_sq_norms, functorch_sq_norms, atol=1e-4, rtol=1e-3
        ):
            print("\nSUCCESS: Opacus loop matches functorch computation!")
        else:
            print(
                f"\nMISMATCH: Max diff = {(total_opacus_sq_norms - functorch_sq_norms).abs().max()}"
            )

    # with BatchStatSnapshot(bn_model_fast, x_bn):
    #     # Get the dimension of the last layer
    #     output_dim = bn_model_fast[-1].out_features

    #     mb_grad_norms = 0
    #     gs_bn_model_norms = 0

    #     # Loop over each output dimension
    #     for dim in range(output_dim):
    #         print(f"\nVerifying norms for output dimension {dim}...")
    #         # Create a modified model that only outputs the current dimension

    #         single_output_model = SingleOutputModel(bn_model_fast, dim)
    #         gs_single_output_model = GhostNormFastGradientClipping(single_output_model, strict=False)
    #         gs_single_output_model.zero_grad()
    #         out = gs_single_output_model(x_bn)
    #         print(f"Output shape: {out.shape}")
    #         out = out.sum()
    #         out.backward()
    #         opacus_sq_norms = gs_single_output_model.get_squared_norm_sample()
    #         gs_bn_model_norms += opacus_sq_norms

    #         # Compute the gradient directly on the microbatch model
    #         mb_single_output_model = SingleOutputModel(mb_model, dim)
    #         mb_grads = compute_microbatch_gradients(mb_single_output_model, x_bn)
    #         mb_sq_norms = torch.zeros_like(opacus_sq_norms)
    #         for name, mb_grad in mb_grads.items():
    #             mb_sq_norms += mb_grad.reshape(mb_grad.shape[0], -1).pow(2).sum(dim=1)
    #         mb_grad_norms += mb_sq_norms

    #         # Compare the gradient magnitudes
    #         print(f"Opacus norms: {gs_bn_model_norms}")
    #         # print(f"Microbatch norms: {mb_grad_norms}")


if __name__ == "__main__":
    test_batch_norm_ghost_norm()
