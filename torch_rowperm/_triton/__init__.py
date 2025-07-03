"""Triton-accelerated row permutation implementations."""

from .row_perm import (
    permute_rows_triton,
    permute_rows_triton_,
    benchmark_vs_native_cuda,
)

__all__ = [
    "permute_rows_triton",
    "permute_rows_triton_",
    "benchmark_vs_native_cuda",
]

# Make these aliases available to be imported as the default permute_rows by __init__.py
permute_rows = permute_rows_triton
permute_rows_ = permute_rows_triton_ 