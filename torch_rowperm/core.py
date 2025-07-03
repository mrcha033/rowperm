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


def is_row_contiguous(tensor):
    """Check if tensor has contiguous rows even if not fully contiguous."""
    if tensor.is_contiguous():
        return True
    
    # For 2D tensors, check if rows are contiguous
    if tensor.dim() == 2:
        # If stride[0] >= size[1] * stride[1], then rows are contiguous
        return tensor.stride(0) >= tensor.size(1) * tensor.stride(1)
    
    # For higher dimensions, more complex check
    if tensor.dim() > 2:
        # Check if all dimensions except dim 0 are contiguous
        inner_size = 1
        for i in range(tensor.dim() - 1, 0, -1):
            if tensor.stride(i) != inner_size:
                return False
            inner_size *= tensor.size(i)
        
        # Check if stride[0] is at least the size of a row
        return tensor.stride(0) >= inner_size
    
    return False


class RowPermuteFn(Function):
    """Autograd function for row permutation."""
    
    @staticmethod
    def forward(ctx, input, indices, inplace=False):
        # Save for backward
        ctx.save_for_backward(indices)
        ctx.inplace = inplace
        
        # Check if the tensor is viable for in-place operation
        if inplace:
            if not is_row_contiguous(input):
                raise ValueError(
                    "In-place row permutation requires tensor with contiguous rows. "
                    "Consider using input.contiguous() first or set inplace=False."
                )
            ctx.mark_dirty(input)
        
        if input.is_cuda and CUDA_AVAILABLE:
            if inplace:
                return _row_permute_inplace_cuda(input, indices)
            else:
                return _row_permute_cuda(input, indices)
        else:
            # CPU fallback
            if inplace:
                # For in-place, we need a temporary buffer to avoid overwriting data
                # before we read it
                if input.is_contiguous():
                    # Safe path: Create temp, copy to input
                    temp = input.clone()
                    input.copy_(temp[indices])
                    return input
                else:
                    # This should never happen due to check above
                    warnings.warn("Unexpected non-contiguous tensor in in-place operation")
                    temp = input.clone()
                    result = temp[indices]
                    input.copy_(result)
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
    
    # Validate indices values are within range
    if torch.min(indices).item() < 0 or torch.max(indices).item() >= input.size(0):
        raise ValueError(f"Indices values must be between 0 and {input.size(0)-1}")
    
    # Check for duplicates in indices
    if indices.unique().size(0) != indices.size(0):
        warnings.warn(
            "Duplicate indices detected. This will result in undefined behavior "
            "for in-place operations as some rows will be overwritten multiple times."
        )
    
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


def benchmark_comparison(sizes=[(1000, 512), (10000, 256), (100000, 128)], 
                         dtype=torch.float32, device='cuda'):
    """Run a benchmark comparing native PyTorch vs torch_rowperm.
    
    Returns:
        DataFrame with benchmark results
    """
    try:
        import pandas as pd
        import time
    except ImportError:
        print("pandas is required for benchmarking")
        return
    
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    results = []
    
    for rows, cols in sizes:
        x = torch.randn(rows, cols, dtype=dtype, device=device)
        idx = torch.randperm(rows, device=device)
        
        # Warm up
        for _ in range(5):
            torch.cuda.synchronize() if device == 'cuda' else None
            _ = x[idx]
            _ = permute_rows(x, idx)
        
        # PyTorch native
        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.perf_counter()
        for _ in range(10):
            y_native = x[idx]
        torch.cuda.synchronize() if device == 'cuda' else None
        native_time = (time.perf_counter() - t0) * 1000 / 10  # ms
        
        # torch_rowperm
        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.perf_counter()
        for _ in range(10):
            y_rowperm = permute_rows(x, idx)
        torch.cuda.synchronize() if device == 'cuda' else None
        rowperm_time = (time.perf_counter() - t0) * 1000 / 10  # ms
        
        # Validate results
        torch.testing.assert_close(y_native, y_rowperm)
        
        speedup = native_time / rowperm_time if rowperm_time > 0 else float('inf')
        
        results.append({
            'Rows': rows,
            'Cols': cols,
            'Native (ms)': round(native_time, 2),
            'RowPerm (ms)': round(rowperm_time, 2),
            'Speedup': round(speedup, 2)
        })
    
    return pd.DataFrame(results) 