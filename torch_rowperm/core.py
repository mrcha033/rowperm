import torch
from torch.autograd import Function

try:
    from torch_rowperm._C import row_permute as _row_permute_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class RowPermuteFn(Function):
    """Autograd function for row permutation."""
    
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        
        if input.is_cuda and CUDA_AVAILABLE:
            return _row_permute_cuda(input, indices)
        else:
            # CPU fallback
            return input[indices]
    
    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        
        # Create inverse permutation
        inv_indices = torch.empty_like(indices)
        inv_indices[indices] = torch.arange(
            indices.size(0), dtype=indices.dtype, device=indices.device
        )
        
        if grad_output.is_cuda and CUDA_AVAILABLE:
            grad_input = _row_permute_cuda(grad_output, inv_indices)
        else:
            grad_input = grad_output[inv_indices]
        
        return grad_input, None


def permute_rows(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Permute rows of a tensor according to given indices.
    
    Args:
        input: Input tensor of shape (N, ...)
        indices: Permutation indices of shape (N,)
    
    Returns:
        Permuted tensor with same shape as input
    
    Example:
        >>> x = torch.randn(4, 3)
        >>> idx = torch.tensor([2, 0, 3, 1])
        >>> y = permute_rows(x, idx)
        >>> assert torch.equal(y[0], x[2])
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Expected torch.Tensor, got {type(input)}")
    
    if not torch.is_tensor(indices):
        raise TypeError(f"Expected torch.Tensor for indices, got {type(indices)}")
    
    if input.dim() < 1:
        raise ValueError("Input must be at least 1D")
    
    if indices.dim() != 1:
        raise ValueError("Indices must be 1D")
    
    if indices.size(0) != input.size(0):
        raise ValueError(
            f"Indices length ({indices.size(0)}) must match "
            f"input's first dimension ({input.size(0)})"
        )
    
    return RowPermuteFn.apply(input, indices) 