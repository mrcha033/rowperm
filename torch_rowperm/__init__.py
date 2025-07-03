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
    from ._triton import permute_rows as _permute_rows_triton
    from ._triton import permute_rows_ as _permute_rows_inplace_triton
    from ._triton import permute_rows_triton, permute_rows_triton_
    from ._triton import benchmark_vs_native_cuda
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Default implementation priority:
# 1. Triton (if available)
# 2. CUDA (if available)
# 3. PyTorch native (fallback)

if HAS_TRITON:
    # Use Triton as primary implementation
    permute_rows = _permute_rows_triton
    permute_rows_ = _permute_rows_inplace_triton
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
    __all__.extend(["permute_rows_triton", "permute_rows_triton_", "benchmark_vs_native_cuda"])

if HAS_CUDA and HAS_TRITON:
    # Also export CUDA-specific functions if both are available
    __all__.extend(["permute_rows_cuda", "permute_rows_cuda_"])
    permute_rows_cuda = _permute_rows_cuda
    permute_rows_cuda_ = _permute_rows_inplace_cuda 