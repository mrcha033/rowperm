"""torch_rowperm - Fast row permutation operations for PyTorch tensors."""

__version__ = "0.1.0"

from .core import benchmark_comparison

# Try to import CUDA implementation first
try:
    from ._C import row_permute as _permute_rows_cuda
    from ._C import row_permute_inplace as _permute_rows_inplace_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# Try to import Triton implementation
try:
    import triton
    # Check if at least Triton 2.0.0 is available
    triton_version = getattr(triton, "__version__", "0.0.0")
    major, minor = map(int, triton_version.split(".")[:2] + ["0"]*(2 - len(triton_version.split(".")[:2])))
    
    if major >= 2:
        from ._triton.row_perm import (
            permute_rows_triton,
            permute_rows_triton_,
            benchmark_vs_native as benchmark_triton_vs_native
        )
        HAS_TRITON = True
    else:
        HAS_TRITON = False
except ImportError:
    HAS_TRITON = False

# Default implementation priority:
# 1. Triton (if available)
# 2. CUDA (if available)
# 3. PyTorch native (fallback)

if HAS_TRITON:
    # Use Triton as primary implementation
    permute_rows = permute_rows_triton
    permute_rows_ = permute_rows_triton_
elif HAS_CUDA:
    # Use CUDA implementation
    permute_rows = _permute_rows_cuda
    permute_rows_ = _permute_rows_inplace_cuda
else:
    # Fallback to PyTorch native (defined in core.py)
    from .core import permute_rows, permute_rows_

# Export all implementations if available
__all__ = ["permute_rows", "permute_rows_", "benchmark_comparison", "__version__", "HAS_CUDA", "HAS_TRITON"]

if HAS_TRITON:
    __all__.extend([
        "permute_rows_triton", 
        "permute_rows_triton_", 
        "benchmark_triton_vs_native"
    ])

if HAS_CUDA and HAS_TRITON:
    # Also export CUDA-specific functions if both are available
    __all__.extend(["permute_rows_cuda", "permute_rows_cuda_"])
    permute_rows_cuda = _permute_rows_cuda
    permute_rows_cuda_ = _permute_rows_inplace_cuda 

# Add function to compare all available implementations
def benchmark_all_implementations(
    sizes=[(1000, 512), (10000, 256), (100000, 128)],
    dtype=None, 
    device='cuda', 
    iterations=20
):
    """Compare all available implementations: Native PyTorch vs CUDA vs Triton.
    
    Args:
        sizes: List of (rows, cols) tuples to benchmark
        dtype: Data type to use (defaults to torch.float32)
        device: Device to run on
        iterations: Number of iterations for timing
        
    Returns:
        dict: Dictionary with benchmark results for each implementation
    """
    import torch
    if dtype is None:
        dtype = torch.float32
        
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Get available implementations
    implementations = []
    
    # Always have native PyTorch
    implementations.append(('native', lambda x, idx: x[idx]))
    
    # CUDA implementation if available
    if HAS_CUDA and device == 'cuda':
        implementations.append(('cuda', _permute_rows_cuda))
    
    # Triton implementation if available
    if HAS_TRITON and device == 'cuda':
        implementations.append(('triton', permute_rows_triton))
    
    # Try to use torch.utils.benchmark if available
    try:
        from torch.utils.benchmark import Timer as BenchmarkTimer
        use_benchmark = True
    except ImportError:
        use_benchmark = False
        import time
        
    results = {}
    
    for size_idx, (rows, cols) in enumerate(sizes):
        size_results = {}
        x = torch.randn(rows, cols, dtype=dtype, device=device)
        idx = torch.randperm(rows, device=device)
        
        # Warmup
        if device == 'cuda':
            torch.cuda.synchronize()
        
        for name, impl_fn in implementations:
            _ = impl_fn(x, idx)
        
        # Get reference output from native PyTorch
        reference = x[idx]
        
        for name, impl_fn in implementations:
            if use_benchmark:
                # Use PyTorch benchmark timer
                timer = BenchmarkTimer(
                    stmt=f"impl_fn(x, idx)",
                    globals={"impl_fn": impl_fn, "x": x, "idx": idx}
                )
                elapsed_time = timer.blocked_autorange().median * 1000  # ms
            else:
                # Manual timing
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iterations):
                    output = impl_fn(x, idx)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed_time = (time.perf_counter() - t0) * 1000 / iterations  # ms
            
            # Verify correctness
            output = impl_fn(x, idx)
            is_correct = torch.allclose(output, reference)
            
            # Store results
            size_results[name] = {
                'time_ms': round(elapsed_time, 2),
                'correct': is_correct
            }
            
            # Calculate speedup compared to native
            if name != 'native':
                native_time = size_results['native']['time_ms']
                speedup = native_time / elapsed_time if elapsed_time > 0 else float('inf')
                size_results[name]['speedup'] = round(speedup, 2)
        
        results[f"{rows}x{cols}"] = size_results
    
    return results

# Add the new function to exports
__all__.append("benchmark_all_implementations") 