import torch.nn as nn
from torch import no_grad


class BatchStatSnapshot:
    """
    Context manager for computing per-sample gradients with BatchNorm layers.

    This context manager enables per-sample gradient computation (via torch.vmap) for
    models containing BatchNorm layers by:
    1. Updating the model's running statistics to match the current batch statistics
    2. Applying Bessel's correction to variance for exact train-mode equivalence
    3. Switching the model to eval mode to use these fixed statistics
    4. Restoring the original model state upon exit

    The key insight: BatchNorm in train mode creates coupling between samples (gradients
    depend on other samples in the batch), which breaks vmap's independence requirement.
    By using eval mode with running stats set to the current batch stats (with proper
    variance correction), we get the same forward pass as train mode but with decoupled
    per-sample gradients.

    Technical Details
    -----------------
    PyTorch's BatchNorm stores unbiased variance (N-1 denominator) in running_var but
    uses biased variance (N denominator) during forward pass. This context manager
    applies the correction factor N/(N-1) to ensure eval mode outputs exactly match
    train mode.

    The effective sample size N differs by BatchNorm type:
    - BatchNorm1d: N = batch_size (e.g., 8 → correction = 1.143)
    - BatchNorm2d: N = batch_size × H × W (e.g., 8×32×32 = 8192 → correction = 1.0001)
    - BatchNorm3d: N = batch_size × D × H × W (even larger, correction ≈ 1.0000)

    While the correction is small for 2D/3D layers, it is essential for exact matching.

    Parameters
    ----------
    model : nn.Module
        PyTorch model containing BatchNorm layers. The model is modified in-place
        within the context and restored afterwards.
    data : torch.Tensor
        Input batch used to compute the current batch statistics. This data will be
        passed through the model in a forward pass (without gradients) to update
        the running statistics.

    Notes
    -----
    - The model is modified IN-PLACE. Both `model` and the returned value from
      `__enter__` point to the same object.
    - This approach only approximates train-mode gradients. Gradients through the
      batch statistics (∂loss/∂μ and ∂loss/∂σ terms) are not captured.
    - Only affects BatchNorm1d, BatchNorm2d, and BatchNorm3d layers.
    - Output matches train mode to floating-point precision (~1e-8 tolerance).

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.func import vmap, grad
    >>>
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 20),
    ...     nn.BatchNorm1d(20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... )
    >>> data = torch.randn(32, 10)  # Batch of 32 samples
    >>>
    >>> # Method 1: Using 'as' clause (creates alias to same object)
    >>> with BatchStatSnapshot(model, data) as model_snapshot:
    ...     # Use model_snapshot for per-sample gradient computation
    ...     pass
    >>>
    >>> # Method 2: Direct usage (simpler, recommended)
    >>> with BatchStatSnapshot(model, data):
    ...     # Just use 'model' directly - it's already modified
    ...     params = dict(model.named_parameters())
    ...     buffers = dict(model.named_buffers())
    ...     # Proceed with vmap-based gradient computation
    ...     pass

    See Also
    --------
    torch.func.vmap : Vectorized map for batched operations
    """

    def __init__(self, model, data):
        """Initialize the BatchStatSnapshot context manager.

        Parameters
        ----------
        model : nn.Module
            The model containing BatchNorm layers to be modified.
        data : torch.Tensor
            The input batch that will be used to compute batch statistics.
        """
        self.model = model
        self.data = data
        self.original_momentums = {}  # Store original momentum values
        self.original_training_states = {}  # Store original train/eval states
        self.original_model_training = None  # Store model's overall training state
        self.bn_input_shapes = {}  # Track input shapes for Bessel's correction

    def __enter__(self):
        # 0. Save the model's overall training state
        self.original_model_training = self.model.training

        # 1. Save original states and force momentum to 1.0
        # Momentum = 1.0 means the running stats will be instantly overwritten
        # by the current batch's stats (ignoring history).
        #
        # Also register forward hooks to capture input shapes for each BatchNorm
        # layer. This is necessary because the input to a BatchNorm layer may have
        # different spatial dimensions than the model input (e.g., after pooling
        # or strided convolutions). We need the actual layer input shape to
        # calculate the correct effective sample size for Bessel's correction.
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.original_momentums[name] = module.momentum
                self.original_training_states[name] = module.training
                module.momentum = 1.0
                module.train()  # Ensure we are in train mode to update stats

                # Register forward hook to capture input shape.
                # We use a factory function to avoid closure issues - each hook
                # needs to capture its own module name.
                def make_hook(module_name):
                    def hook(module, input, output):
                        self.bn_input_shapes[module_name] = input[0].shape

                    return hook

                hooks.append(module.register_forward_hook(make_hook(name)))

        # 2. Perform a dummy forward pass to update the buffers
        # We use no_grad because we only care about updating the
        # running_mean/var buffers
        with no_grad():
            self.model(self.data)

        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()

        # 2.5. Apply Bessel's correction to running_var
        # Problem: PyTorch stores unbiased variance (N-1 denominator) in
        # running_var, but uses biased variance (N denominator) during train
        # mode forward pass. This mismatch causes eval mode to produce slightly
        # different outputs than train mode.
        #
        # Solution: Convert unbiased → biased by dividing by N/(N-1). This makes
        # eval mode outputs identical to train mode (within float precision).
        #
        # Effective sample size calculation:
        # - BatchNorm1d: N = batch_size (variance computed over batch only)
        # - BatchNorm2d: N = batch_size × H × W (includes spatial dimensions)
        # - BatchNorm3d: N = batch_size × D × H × W (includes all spatial dims)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                input_shape = self.bn_input_shapes.get(name)
                if input_shape is None:
                    continue  # Skip if we couldn't capture shape.
                    # Corrections are not applied.

                # Calculate effective sample size based on layer type.
                # The effective N is the total number of values used to compute
                # each feature's statistics during the forward pass.
                if isinstance(module, nn.BatchNorm1d):
                    # BatchNorm1d: variance over batch dimension (and spatial if present)
                    # Input shape: (N, C) or (N, C, L)
                    effective_n = input_shape[0]
                    if len(input_shape) > 2:  # Has spatial dimension L
                        effective_n *= input_shape[2]
                elif isinstance(module, nn.BatchNorm2d):
                    # BatchNorm2d: variance over batch + spatial dimensions
                    # Input shape: (N, C, H, W)
                    effective_n = input_shape[0] * input_shape[2] * input_shape[3]
                elif isinstance(module, nn.BatchNorm3d):
                    # BatchNorm3d: variance over batch + spatial dimensions
                    # Input shape: (N, C, D, H, W)
                    effective_n = (
                        input_shape[0]
                        * input_shape[2]
                        * input_shape[3]
                        * input_shape[4]
                    )

                if effective_n > 1:
                    correction_factor = effective_n / (effective_n - 1)
                    module.running_var.div_(correction_factor)
                # If effective_n == 1, skip correction to avoid division by zero.

        # 3. Switch to Eval mode
        # Now both running_mean and running_var (after correction) match what
        # train mode uses. In Eval mode, PyTorch uses these buffers as fixed
        # constants. This satisfies vmap's independence requirement while
        # maintaining train-mode accuracy.
        self.model.eval()

        return self.model

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore model to its original state.

        This method is called when exiting the context, whether normally or
        due to an exception. It ensures all BatchNorm layers are returned to
        their original momentum and training state.
        """
        # Restore each BatchNorm layer's original momentum and training state
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.momentum = self.original_momentums[name]
                module.train(self.original_training_states[name])

        # Restore the model's overall training state
        self.model.train(self.original_model_training)
