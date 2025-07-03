import time
import pytest
import torch
import torch_rowperm as rp


@pytest.mark.gpu
@pytest.mark.benchmark
class TestPerformance:
    """Benchmark row permutation performance."""
    
    def test_speed_vs_native(self):
        """Compare speed against native PyTorch indexing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Test different sizes
        sizes = [(1000, 512), (10000, 256), (100000, 128)]
        
        for num_rows, row_size in sizes:
            x = torch.randn(num_rows, row_size, device="cuda")
            idx = torch.randperm(num_rows, device="cuda")
            
            # Warmup
            for _ in range(10):
                _ = rp.permute_rows(x, idx)
                _ = x[idx]
            
            torch.cuda.synchronize()
            
            # Time our implementation
            start = time.perf_counter()
            for _ in range(100):
                y1 = rp.permute_rows(x, idx)
            torch.cuda.synchronize()
            our_time = time.perf_counter() - start
            
            # Time native PyTorch
            start = time.perf_counter()
            for _ in range(100):
                y2 = x[idx]
            torch.cuda.synchronize()
            native_time = time.perf_counter() - start
            
            print(f"\nSize {num_rows}x{row_size}:")
            print(f"  Our impl: {our_time:.4f}s")
            print(f"  Native:   {native_time:.4f}s")
            print(f"  Speedup:  {native_time/our_time:.2f}x")
            
            # Verify correctness
            torch.testing.assert_close(y1, y2)
    
    def test_memory_efficiency(self):
        """Test memory usage patterns."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(50000, 512, device="cuda")
        idx = torch.randperm(50000, device="cuda")
        
        start_mem = torch.cuda.memory_allocated()
        y = rp.permute_rows(x, idx)
        peak_mem = torch.cuda.max_memory_allocated()
        
        # Should only allocate output tensor
        expected_mem = x.numel() * x.element_size()
        actual_additional = peak_mem - start_mem
        
        print(f"\nMemory usage:")
        print(f"  Expected: {expected_mem / 1024**2:.1f} MB")
        print(f"  Actual:   {actual_additional / 1024**2:.1f} MB")
        
        # Allow some overhead but not double
        assert actual_additional < expected_mem * 1.5 