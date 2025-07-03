"""Speed tests for row permutation operations."""

import pytest
import torch
import time

try:
    import torch_rowperm as rp
    HAS_ROWPERM = True
except ImportError:
    HAS_ROWPERM = False


@pytest.mark.parametrize("shape", [
    (1000, 128),
    (10000, 256),
    (100000, 128),
])
def test_benchmark_speedup(shape):
    """Test that our implementation is faster than native PyTorch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    if not HAS_ROWPERM:
        pytest.skip("torch_rowperm not available")
    
    device = "cuda"
    iterations = 10
    
    # Create tensors
    x = torch.randn(shape, device=device)
    idx = torch.randperm(shape[0], device=device)
    
    # Warmup
    _ = x[idx]
    _ = rp.permute_rows(x, idx)
    torch.cuda.synchronize()
    
    # Time native implementation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        y_native = x[idx]
    torch.cuda.synchronize()
    native_time = (time.perf_counter() - start) * 1000 / iterations  # ms
    
    # Time our implementation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        y_ours = rp.permute_rows(x, idx)
    torch.cuda.synchronize()
    our_time = (time.perf_counter() - start) * 1000 / iterations  # ms
    
    # Calculate speedup
    speedup = native_time / our_time
    
    # We expect at least 2x speedup
    assert speedup >= 2.0, f"Speedup was only {speedup:.2f}x for shape {shape}"
    
    print(f"Shape {shape}: Native: {native_time:.2f} ms, Ours: {our_time:.2f} ms, Speedup: {speedup:.2f}x")


@pytest.mark.parametrize("impl", ["cuda", "triton", "native"])
def test_implementation_comparison(impl):
    """Test and compare all available implementations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    if not HAS_ROWPERM:
        pytest.skip("torch_rowperm not available")
    
    if impl == "triton" and not rp.HAS_TRITON:
        pytest.skip("Triton not available")
    
    if impl == "cuda" and not rp.HAS_CUDA:
        pytest.skip("CUDA implementation not available")
    
    device = "cuda"
    shapes = [
        (1000, 128),    # Small
        (10000, 256),   # Medium
        (100000, 128),  # Large
    ]
    iterations = 10
    
    for shape in shapes:
        # Create tensors
        x = torch.randn(shape, device=device)
        idx = torch.randperm(shape[0], device=device)
        
        # Select implementation
        if impl == "native":
            fn = lambda x, idx: x[idx]
        elif impl == "cuda":
            fn = rp.permute_rows_cuda if hasattr(rp, "permute_rows_cuda") else rp.permute_rows
        else:  # triton
            fn = rp.permute_rows_triton
        
        # Warmup
        _ = fn(x, idx)
        torch.cuda.synchronize()
        
        # Time implementation
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            y = fn(x, idx)
        torch.cuda.synchronize()
        elapsed_time = (time.perf_counter() - start) * 1000 / iterations  # ms
        
        print(f"Shape {shape}, {impl}: {elapsed_time:.2f} ms")


def test_comprehensive_benchmark():
    """Run the comprehensive benchmark function if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    if not HAS_ROWPERM:
        pytest.skip("torch_rowperm not available")
    
    if not hasattr(rp, "benchmark_all_implementations"):
        pytest.skip("benchmark_all_implementations not available")
    
    # Run benchmark with common sizes
    results = rp.benchmark_all_implementations(
        sizes=[
            (1000, 128),
            (10000, 256),
            (100000, 128),
        ],
        iterations=5  # Keep low for CI
    )
    
    # Print results
    for size, size_results in results.items():
        print(f"\nSize: {size}")
        for impl, data in size_results.items():
            if impl == "native":
                print(f"  {impl}: {data['time_ms']:.2f} ms")
            else:
                print(f"  {impl}: {data['time_ms']:.2f} ms, speedup: {data.get('speedup', 0):.2f}x")
    
    # Make sure all implementations give correct results
    for size_results in results.values():
        for impl, data in size_results.items():
            assert data["correct"], f"Implementation {impl} produced incorrect results"


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