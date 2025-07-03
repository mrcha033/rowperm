"""Correctness tests for row permutation operations."""

import pytest
import torch

try:
    import torch_rowperm as rp
    HAS_ROWPERM = True
except ImportError:
    HAS_ROWPERM = False


@pytest.mark.parametrize("dtype", [
    torch.float32, 
    torch.float16, 
    torch.bfloat16
])
@pytest.mark.parametrize("shape", [
    (10, 20),
    (100, 128),
    (1000, 256),
    (5000, 512),
])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_permute_rows_correctness(dtype, shape, device):
    """Test if permute_rows produces the same result as native PyTorch indexing."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    if not HAS_ROWPERM:
        pytest.skip("torch_rowperm not available")
        
    # Skip unsupported half precision on CPU
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(f"{dtype} not supported on CPU")
        
    # Create test tensors
    x = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    idx = torch.randperm(shape[0], device=device)
    
    # Reference implementation
    y_ref = x[idx]
    
    # Test CUDA implementation
    if hasattr(rp, 'permute_rows_cuda') and device == 'cuda':
        y_cuda = rp.permute_rows_cuda(x, idx)
        assert torch.allclose(y_ref, y_cuda, rtol=1e-2, atol=1e-2)
        
        # Test in-place version
        x_inplace = x.clone()
        rp.permute_rows_cuda_(x_inplace, idx)
        assert torch.allclose(y_ref, x_inplace, rtol=1e-2, atol=1e-2)
    
    # Test Triton implementation
    if hasattr(rp, 'permute_rows_triton') and rp.HAS_TRITON and device == 'cuda':
        y_triton = rp.permute_rows_triton(x, idx)
        assert torch.allclose(y_ref, y_triton, rtol=1e-2, atol=1e-2)
        
        # Test in-place version
        x_inplace = x.clone()
        rp.permute_rows_triton_(x_inplace, idx)
        assert torch.allclose(y_ref, x_inplace, rtol=1e-2, atol=1e-2)
    
    # Test default implementation (should choose the best available)
    y_impl = rp.permute_rows(x, idx)
    assert torch.allclose(y_ref, y_impl, rtol=1e-2, atol=1e-2)
    
    # Test in-place version
    x_inplace = x.clone()
    rp.permute_rows_(x_inplace, idx)
    assert torch.allclose(y_ref, x_inplace, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [
    (100, 128), 
    (1000, 256)
])
def test_autograd_correctness(shape):
    """Test if autograd works correctly with permute_rows."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    if not HAS_ROWPERM:
        pytest.skip("torch_rowperm not available")
    
    device = "cuda"
    
    # Create test tensors
    x = torch.randn(shape, device=device, requires_grad=True)
    idx = torch.randperm(shape[0], device=device)
    
    # Forward pass with native PyTorch
    y_ref = x[idx]
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_ref = x.grad.clone()
    
    # Reset gradients
    x.grad = None
    
    # Forward pass with default implementation
    y_impl = rp.permute_rows(x, idx)
    loss_impl = y_impl.sum()
    loss_impl.backward()
    grad_impl = x.grad.clone()
    
    # Check if gradients match
    assert torch.allclose(grad_ref, grad_impl, rtol=1e-5, atol=1e-5)
    
    # Test Triton implementation if available
    if hasattr(rp, 'permute_rows_triton') and rp.HAS_TRITON:
        # Reset gradients
        x.grad = None
        
        # Forward pass with Triton
        y_triton = rp.permute_rows_triton(x, idx)
        loss_triton = y_triton.sum()
        loss_triton.backward()
        grad_triton = x.grad.clone()
        
        # Check if gradients match
        assert torch.allclose(grad_ref, grad_triton, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_edge_cases(dtype):
    """Test edge cases for permutation operations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    if not HAS_ROWPERM:
        pytest.skip("torch_rowperm not available")
    
    device = "cuda"
    
    # Test 1: Single row
    x = torch.randn(1, 128, dtype=dtype, device=device)
    idx = torch.tensor([0], device=device)
    y_ref = x[idx]
    y_impl = rp.permute_rows(x, idx)
    assert torch.allclose(y_ref, y_impl, rtol=1e-2, atol=1e-2)
    
    # Test 2: Identity permutation
    x = torch.randn(100, 128, dtype=dtype, device=device)
    idx = torch.arange(100, device=device)
    y_ref = x[idx]
    y_impl = rp.permute_rows(x, idx)
    assert torch.allclose(y_ref, y_impl, rtol=1e-2, atol=1e-2)
    
    # Test 3: Reverse permutation
    x = torch.randn(100, 128, dtype=dtype, device=device)
    idx = torch.arange(100, device=device).flip(0)
    y_ref = x[idx]
    y_impl = rp.permute_rows(x, idx)
    assert torch.allclose(y_ref, y_impl, rtol=1e-2, atol=1e-2)
    
    # Test 4: Tiny rows
    x = torch.randn(100, 3, dtype=dtype, device=device)
    idx = torch.randperm(100, device=device)
    y_ref = x[idx]
    y_impl = rp.permute_rows(x, idx)
    assert torch.allclose(y_ref, y_impl, rtol=1e-2, atol=1e-2)
    
    # Test 5: Odd-sized rows
    x = torch.randn(100, 129, dtype=dtype, device=device)
    idx = torch.randperm(100, device=device)
    y_ref = x[idx]
    y_impl = rp.permute_rows(x, idx)
    assert torch.allclose(y_ref, y_impl, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("impl", ["default", "cuda", "triton"])
def test_api_consistency(impl):
    """Test that the API is consistent across implementations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    if not HAS_ROWPERM:
        pytest.skip("torch_rowperm not available")
    
    if impl == "triton" and not rp.HAS_TRITON:
        pytest.skip("Triton not available")
    
    if impl == "cuda" and not rp.HAS_CUDA:
        pytest.skip("CUDA implementation not available")
    
    device = "cuda"
    x = torch.randn(100, 128, device=device)
    idx = torch.randperm(100, device=device)
    
    # Select implementation
    if impl == "default":
        fn = rp.permute_rows
        fn_inplace = rp.permute_rows_
    elif impl == "cuda":
        fn = rp.permute_rows_cuda
        fn_inplace = rp.permute_rows_cuda_
    else:  # triton
        fn = rp.permute_rows_triton
        fn_inplace = rp.permute_rows_triton_
    
    # Test basic functionality
    y = fn(x, idx)
    assert y.shape == x.shape
    
    # Test in-place functionality
    x_inplace = x.clone()
    y_inplace = fn_inplace(x_inplace, idx)
    assert torch.allclose(y, x_inplace)
    assert id(y_inplace) == id(x_inplace)  # Should return self 