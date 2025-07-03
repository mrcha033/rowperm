import torch
from torch.autograd import Function
import platform
import warnings

try:
    from torch_rowperm._C import row_permute as _row_permute_cuda
    from torch_rowperm._C import row_permute_inplace as _row_permute_inplace_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    if torch.cuda.is_available() and platform.system() in ["Linux", "Windows"]:
        warnings.warn(
            "CUDA is available but torch_rowperm CUDA extension could not be loaded. "
            "Falling back to native PyTorch implementation. "
            "For better performance, please reinstall torch_rowperm with CUDA support."
        )


class RowPermuteFn(Function):
    """Autograd function for row permutation."""
    
    @staticmethod
    def forward(ctx, input, indices, inplace=False):
        ctx.save_for_backward(indices)
        ctx.inplace = inplace
        ctx.mark_dirty(input) if inplace else None
        
        if input.is_cuda and CUDA_AVAILABLE:
            if inplace:
                return _row_permute_inplace_cuda(input, indices)
            else:
                return _row_permute_cuda(input, indices)
        else:
            # CPU fallback
            if inplace:
                # For in-place, we need to create a temporary copy
                temp = input.clone()
                input.copy_(temp[indices])
                return input
            else:
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
        
        return grad_input, None, None  # None for indices grad and inplace flag


def permute_rows(input: torch.Tensor, indices: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Permute rows of a tensor according to given indices.
    
    Args:
        input: Input tensor of shape (N, ...)
        indices: Permutation indices of shape (N,)
        inplace: Whether to perform the operation in-place (default: False)
    
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
    
    # For in-place operations, make sure input is contiguous
    if inplace and not input.is_contiguous():
        input = input.contiguous()
    
    return RowPermuteFn.apply(input, indices, inplace)


def permute_rows_(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """In-place version of permute_rows.
    
    Args:
        input: Input tensor of shape (N, ...)
        indices: Permutation indices of shape (N,)
    
    Returns:
        Input tensor with permuted rows (modified in-place)
    """
    return permute_rows(input, indices, inplace=True) 