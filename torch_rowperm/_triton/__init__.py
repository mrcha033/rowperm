"""Triton-based implementations for row permutation."""

try:
    from .row_perm import (
        permute_rows_triton,
        permute_rows_triton_,
        benchmark_vs_native
    )
    
    __all__ = [
        "permute_rows_triton",
        "permute_rows_triton_",
        "benchmark_vs_native"
    ]
    
except ImportError:
    # If Triton is not available, provide error functions
    def _not_available(*args, **kwargs):
        raise ImportError(
            "Triton implementation not available. "
            "Install with pip install 'torch_rowperm[triton]'"
        )
    
    permute_rows_triton = _not_available
    permute_rows_triton_ = _not_available
    benchmark_vs_native = _not_available
    
    __all__ = [
        "permute_rows_triton",
        "permute_rows_triton_",
        "benchmark_vs_native"
    ]

# Make these aliases available to be imported as the default permute_rows by __init__.py
permute_rows = permute_rows_triton
permute_rows_ = permute_rows_triton_ 