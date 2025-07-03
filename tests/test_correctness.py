import pytest
import torch
import torch_rowperm as rp


class TestCorrectness:
    """Test correctness of row permutation operations."""
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_basic_permutation(self, device, dtype):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Simple test case
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
                        device=device, dtype=dtype)
        idx = torch.tensor([2, 0, 1], device=device)
        
        y = rp.permute_rows(x, idx)
        
        expected = torch.tensor([[7, 8, 9], [1, 2, 3], [4, 5, 6]], 
                               device=device, dtype=dtype)
        torch.testing.assert_close(y, expected)
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_idempotent(self, device):
        """Test that permuting and inverse permuting returns original."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        x = torch.randn(128, 512, device=device)
        idx = torch.randperm(128, device=device)
        
        y = rp.permute_rows(x.clone(), idx)
        z = rp.permute_rows(y.clone(), torch.argsort(idx))
        
        torch.testing.assert_close(x, z, rtol=0, atol=0)
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_gradient_flow(self, device):
        """Test that gradients flow correctly through permutation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        x = torch.randn(32, 64, device=device, requires_grad=True)
        idx = torch.randperm(32, device=device)
        
        y = rp.permute_rows(x, idx)
        loss = y.sum()
        loss.backward()
        
        # Gradient should be all ones, but permuted back
        expected_grad = torch.ones_like(x)
        torch.testing.assert_close(x.grad, expected_grad)
    
    def test_error_handling(self):
        """Test error cases."""
        x = torch.randn(10, 5)
        
        # Wrong index length
        with pytest.raises(ValueError, match="Indices length"):
            rp.permute_rows(x, torch.tensor([0, 1]))
        
        # Wrong index dimension
        with pytest.raises(ValueError, match="Indices must be 1D"):
            rp.permute_rows(x, torch.tensor([[0, 1]]))
        
        # Non-tensor input
        with pytest.raises(TypeError):
            rp.permute_rows([1, 2, 3], torch.tensor([0, 1, 2]))
    
    @pytest.mark.gpu
    def test_large_tensor(self):
        """Test with large tensors to catch memory issues."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        x = torch.randn(10000, 1024, device="cuda")
        idx = torch.randperm(10000, device="cuda")
        
        y = rp.permute_rows(x, idx)
        
        # Check a few random elements
        for i in torch.randint(0, 10000, (10,)):
            torch.testing.assert_close(y[i], x[idx[i]]) 