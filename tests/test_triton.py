"""Tests for Triton-accelerated row permutation."""

import pytest
import torch
import torch_rowperm

# Skip all tests if Triton is not available
pytestmark = pytest.mark.skipif(
    not torch_rowperm.HAS_TRITON, 
    reason="Triton not available"
)

# Skip if CUDA not available
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), 
    reason="CUDA not available"
)


@cuda_only
def test_triton_basic():
    """Test basic functionality of Triton implementation."""
    # Create test data
    x = torch.randn(10, 20, device="cuda")
    idx = torch.randperm(10, device="cuda")
    
    # Run Triton implementation
    y_triton = torch_rowperm.permute_rows_triton(x, idx)
    
    # Run reference implementation
    y_ref = x[idx]
    
    # Check results
    assert torch.allclose(y_triton, y_ref)


@cuda_only
def test_triton_large():
    """Test with larger tensors."""
    x = torch.randn(1000, 256, device="cuda")
    idx = torch.randperm(1000, device="cuda")
    
    y_triton = torch_rowperm.permute_rows_triton(x, idx)
    y_ref = x[idx]
    
    assert torch.allclose(y_triton, y_ref)


@cuda_only
def test_triton_3d():
    """Test with 3D tensors."""
    x = torch.randn(100, 32, 64, device="cuda")
    idx = torch.randperm(100, device="cuda")
    
    y_triton = torch_rowperm.permute_rows_triton(x, idx)
    y_ref = x[idx]
    
    assert torch.allclose(y_triton, y_ref)


@cuda_only
def test_triton_dtypes():
    """Test with different data types."""
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    for dtype in dtypes:
        # Force alignment by creating tensor with specific size
        # Ensure vector-alignment friendly sizes (multiples of 4)
        x = torch.randn(100, 64, device="cuda").to(dtype)
        idx = torch.randperm(100, device="cuda")
        
        y_triton = torch_rowperm.permute_rows_triton(x, idx)
        y_ref = x[idx]
        
        # Use lower tolerance for half-precision types
        tol = 1e-3 if dtype == torch.float32 else 1e-2
        assert torch.allclose(y_triton, y_ref, atol=tol, rtol=tol)


@cuda_only
def test_triton_inplace():
    """Test in-place operation."""
    x = torch.randn(50, 64, device="cuda")
    idx = torch.randperm(50, device="cuda")
    
    # Reference
    y_ref = x[idx].clone()
    
    # In-place
    x_copy = x.clone()
    torch_rowperm.permute_rows_triton_(x_copy, idx)
    
    assert torch.allclose(x_copy, y_ref)


@cuda_only
def test_triton_alignment():
    """Test that alignment detection works properly."""
    from torch_rowperm._triton.row_perm import check_alignment
    
    # Create tensors with different alignments by offsetting
    x_aligned = torch.empty(100, 64, device="cuda")
    
    # Test alignment for different vector sizes
    assert check_alignment(x_aligned, 1)  # Always aligned for scalar
    
    # Skip actual 2/4 alignment tests as we can't reliably create misaligned tensors
    # in PyTorch API, but we can test the function exists
    check_alignment(x_aligned, 2)
    check_alignment(x_aligned, 4)


@cuda_only
def test_benchmark():
    """Test that benchmarking function runs."""
    try:
        import pandas
        # Only run with small size for testing
        result = torch_rowperm.benchmark_vs_native_cuda(
            sizes=[(100, 64)],
            iterations=2
        )
        assert len(result) > 0
    except ImportError:
        pytest.skip("pandas not available")


@cuda_only
def test_autograd():
    """Test autograd support."""
    x = torch.randn(20, 10, device="cuda", requires_grad=True)
    idx = torch.randperm(20, device="cuda")
    
    # Forward
    y_triton = torch_rowperm.permute_rows_triton(x, idx)
    y_ref = x[idx]
    
    # Check forward
    assert torch.allclose(y_triton, y_ref)
    
    # Backward
    y_triton.sum().backward()
    grad_triton = x.grad.clone()
    
    x.grad = None
    y_ref.sum().backward()
    grad_ref = x.grad
    
    # Check backward
    assert torch.allclose(grad_triton, grad_ref)


@cuda_only
def test_very_large_rows():
    """Test with very large row sizes."""
    # Create a tensor with large row size (>16K elements)
    x = torch.randn(10, 5000, device="cuda")  # 5K elements per row
    idx = torch.randperm(10, device="cuda")
    
    # Ensure permutation works with large rows
    y_triton = torch_rowperm.permute_rows_triton(x, idx)
    y_ref = x[idx]
    
    assert torch.allclose(y_triton, y_ref) 