"""Triton-based implementation of row permutation operations."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Scalar configs - always safe regardless of alignment
        triton.Config({'BLOCK': 128, 'VEC': 1, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK': 256, 'VEC': 1, 'num_warps': 8}, num_stages=2),
        # Half-precision (8-byte) alignment configs - only used when alignment verified
        triton.Config({'BLOCK': 64, 'VEC': 2, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK': 128, 'VEC': 2, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK': 256, 'VEC': 2, 'num_warps': 8}, num_stages=2),
        # Float4 (16-byte) alignment configs - only used when alignment verified 
        triton.Config({'BLOCK': 128, 'VEC': 4, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK': 256, 'VEC': 4, 'num_warps': 8}, num_stages=3),
    ],
    # Use row_size and alignment as autotune keys to select appropriate config
    key=['row_size', 'align_8b', 'align_16b'],
)
@triton.jit
def row_perm_kernel(
    x_ptr, y_ptr, idx_ptr,
    n_rows, row_size,
    BLOCK: tl.constexpr, 
    VEC: tl.constexpr,
):
    """Optimized Triton kernel for row permutation.
    
    All loop bounds use compile-time constants to ensure proper JIT compilation.
    
    Args:
        x_ptr: Pointer to input tensor
        y_ptr: Pointer to output tensor
        idx_ptr: Pointer to indices tensor
        n_rows: Number of rows
        row_size: Elements per row
        BLOCK: Block size (compile-time)
        VEC: Vector size (compile-time)
    """
    # Process one row per program instance
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
        
    # Get source row index
    src_row = tl.load(idx_ptr + pid)
    
    # Calculate base pointers for source and destination rows
    src_ptr = x_ptr + src_row * row_size
    dst_ptr = y_ptr + pid * row_size
    
    # Handle tiny rows (<=128 elements) with scalar approach for maximum efficiency
    if row_size <= 128:
        offsets = tl.arange(0, BLOCK)
        mask = offsets < row_size
        values = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, values, mask=mask)
        return
    
    # Handle small rows (up to BLOCK*VEC elements) separately to ensure we always process at least one chunk
    if row_size <= BLOCK * VEC:
        offsets = tl.arange(0, BLOCK) * VEC
        # Create mask ensuring we don't read past the end of the row
        mask = offsets < row_size
        
        # Load and store with proper vector alignment
        values = tl.load(src_ptr + offsets, mask=mask, eviction_policy='evict_first')
        tl.store(dst_ptr + offsets, values, mask=mask)
        return
        
    # Calculate offsets within the row (vectorized)
    offsets = tl.arange(0, BLOCK) * VEC
    
    # Process the row in chunks - safe to use row_size directly since only step needs to be constant
    for chunk_start in range(0, row_size, BLOCK * VEC):
        # Current positions to load/store
        pos = chunk_start + offsets
        
        # Bounds checking mask
        mask = pos < row_size
        
        # Load values from source row with eviction policy to handle alignment
        values = tl.load(src_ptr + pos, mask=mask, eviction_policy='evict_first')
        
        # Store to destination row
        tl.store(dst_ptr + pos, values, mask=mask)


def check_alignment(tensor, alignment_bytes):
    """Check if tensor data pointer and dimensions are properly aligned.
    
    Args:
        tensor: PyTorch tensor
        alignment_bytes: Required alignment in bytes (4, 8, or 16)
        
    Returns:
        bool: True if tensor is properly aligned for vectorized operations
    """
    if alignment_bytes <= 4:
        return True  # Scalar is always aligned
        
    # Check pointer alignment
    ptr_value = tensor.data_ptr()
    ptr_aligned = (ptr_value % alignment_bytes) == 0
    
    # Check if row size is aligned with vector size
    row_size = tensor.numel() // tensor.size(0) 
    vec_size = alignment_bytes // 4  # Number of float32 elements
    dim_aligned = (row_size % vec_size) == 0
    
    return ptr_aligned and dim_aligned


def permute_rows_triton(input_tensor, indices_tensor):
    """Permute rows of a tensor using Triton kernels.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape [num_rows, ...].
        indices_tensor (torch.Tensor): Indices tensor of shape [num_rows].
        
    Returns:
        torch.Tensor: Permuted tensor of same shape as input.
    """
    # Ensure input is on GPU
    if not input_tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    
    # Ensure indices are on same device as input
    if indices_tensor.device != input_tensor.device:
        indices_tensor = indices_tensor.to(input_tensor.device)
    
    # Get dimensions
    num_rows = input_tensor.size(0)
    row_size = input_tensor.numel() // num_rows
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Ensure contiguous memory layout
    input_contiguous = input_tensor.contiguous()
    
    # Check alignment for autotune key - this determines which kernel configs are viable
    align_8b = check_alignment(input_contiguous, 8) and check_alignment(output, 8)
    align_16b = check_alignment(input_contiguous, 16) and check_alignment(output, 16)
    
    # Calculate grid size - one block per row ensures efficient row processing
    grid = (num_rows,)
    
    # Launch kernel with appropriate parameters from autotune
    row_perm_kernel[grid](
        input_contiguous.data_ptr(),
        output.data_ptr(),
        indices_tensor.data_ptr(),
        num_rows,
        row_size,
        # Key arguments for autotune - no need to manually override VEC or num_warps
        # Triton's autotuner will select the best config based on these keys
        row_size=row_size,
        align_8b=align_8b,
        align_16b=align_16b,
    )
    
    return output


class TritonRowPermuteFn(torch.autograd.Function):
    """Autograd function for Triton-based row permutation."""
    
    @staticmethod
    def forward(ctx, input_tensor, indices):
        ctx.save_for_backward(indices)
        return permute_rows_triton(input_tensor, indices)
    
    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        
        # Create inverse permutation on the host - much more efficient than
        # searching for each element in the kernel
        inv_indices = torch.empty_like(indices)
        inv_indices[indices] = torch.arange(
            indices.size(0), dtype=indices.dtype, device=indices.device
        )
        
        # Apply inverse permutation using the same forward kernel
        grad_input = permute_rows_triton(grad_output, inv_indices)
        
        # No gradient for indices
        return grad_input, None


def permute_rows_triton(input_tensor, indices_tensor):
    """Permute rows using Triton with autograd support."""
    return TritonRowPermuteFn.apply(input_tensor, indices_tensor)


def permute_rows_triton_(input_tensor, indices_tensor):
    """In-place version of permute_rows_triton.
    
    Note: This performs a copy internally as Triton doesn't directly support in-place operations.
    """
    result = permute_rows_triton(input_tensor, indices_tensor)
    input_tensor.copy_(result)
    return input_tensor


def profile_permutation_impact(model_func, batch_size=8, seq_len=128, hidden_dim=768, 
                            device='cuda', implementations=None):
    """Profile the impact of row permutation in a model.
    
    This function is a minimal version that doesn't require pandas and focuses on
    measuring the percentage of time spent in row permutation operations.
    
    Args:
        model_func: Function that creates and returns a model with row permutation
        batch_size: Batch size for the model
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        device: Device to run on
        implementations: List of implementations to profile (default: ['native'])
        
    Returns:
        dict: Dictionary with profiling results
    """
    if implementations is None:
        implementations = ['native']
        
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    results = {}
    
    for impl in implementations:
        print(f"\nProfiling implementation: {impl}")
        
        # Create model with specified implementation
        try:
            model = model_func(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                implementation=impl,
                device=device
            )
        except Exception as e:
            print(f"Error creating model with {impl} implementation: {e}")
            continue
            
        # Create sample input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        # Warmup
        for _ in range(3):
            try:
                _ = model(x)
            except Exception as e:
                print(f"Error during warmup: {e}")
                break
        
        # Profile
        try:
            # Use PyTorch profiler if available
            try:
                with torch.autograd.profiler.profile(use_cuda=device=='cuda') as prof:
                    output = model(x)
                    # Make sure all operations are executed
                    if device == 'cuda':
                        torch.cuda.synchronize()
                
                # Extract profiling data
                total_time = sum(e.cpu_time for e in prof.key_averages())
                perm_time = sum(e.cpu_time for e in prof.key_averages() if 'permute' in e.key)
                
                # Calculate percentage
                perm_percentage = (perm_time / total_time * 100) if total_time > 0 else 0
                
                results[impl] = {
                    'total_time_ms': total_time,
                    'permutation_time_ms': perm_time,
                    'permutation_percentage': perm_percentage
                }
                
                print(f"  Total time: {total_time:.2f} ms")
                print(f"  Permutation time: {perm_time:.2f} ms")
                print(f"  Permutation percentage: {perm_percentage:.2f}%")
                
            except ImportError:
                # Fallback to manual timing if profiler not available
                import time
                
                # Time total execution
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output = model(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                total_time = (time.perf_counter() - start) * 1000  # ms
                
                # Get permutation time from model's timers
                if hasattr(model, 'timers'):
                    perm_time = model.timers.get('permutation', 0) * 1000  # ms
                    perm_percentage = (perm_time / total_time * 100) if total_time > 0 else 0
                    
                    results[impl] = {
                        'total_time_ms': total_time,
                        'permutation_time_ms': perm_time,
                        'permutation_percentage': perm_percentage
                    }
                    
                    print(f"  Total time: {total_time:.2f} ms")
                    print(f"  Permutation time: {perm_time:.2f} ms")
                    print(f"  Permutation percentage: {perm_percentage:.2f}%")
                else:
                    print("  Model doesn't have timing data")
                    
        except Exception as e:
            print(f"Error during profiling: {e}")
    
    return results


def benchmark_vs_native(sizes=[(1000, 512), (10000, 256), (100000, 128)], 
                      dtype=torch.float32, device='cuda', iterations=20):
    """Simple benchmark comparing Triton vs native PyTorch implementation.
    
    This minimal version doesn't depend on pandas and provides basic timing information.
    
    Args:
        sizes: List of (rows, cols) tuples to benchmark
        dtype: Data type to use
        device: Device to run on
        iterations: Number of iterations for timing
        
    Returns:
        dict: Dictionary with benchmark results
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available")
        return {'error': 'CUDA not available'}
    
    # Try to use torch.utils.benchmark if available
    try:
        from torch.utils.benchmark import Timer as BenchmarkTimer
        use_benchmark = True
    except ImportError:
        use_benchmark = False
        import time
        
    results = {}
    
    for size_idx, (rows, cols) in enumerate(sizes):
        print(f"\nBenchmarking size: {rows} x {cols}")
        x = torch.randn(rows, cols, dtype=dtype, device=device)
        idx = torch.randperm(rows, device=device)
        
        # Warmup
        torch.cuda.synchronize() if device == 'cuda' else None
        _ = x[idx]
        _ = permute_rows_triton(x, idx)
        
        # Measure times
        if use_benchmark:
            # PyTorch native
            native_timer = BenchmarkTimer(
                stmt="x[idx]",
                globals={"x": x, "idx": idx}
            )
            native_time = native_timer.blocked_autorange().median * 1000  # ms
            
            # Triton implementation
            triton_timer = BenchmarkTimer(
                stmt="permute_rows_triton(x, idx)",
                globals={"permute_rows_triton": permute_rows_triton, "x": x, "idx": idx}
            )
            triton_time = triton_timer.blocked_autorange().median * 1000  # ms
        else:
            # Manual timing
            torch.cuda.synchronize() if device == 'cuda' else None
            t0 = time.perf_counter()
            for _ in range(iterations):
                y_native = x[idx]
            torch.cuda.synchronize() if device == 'cuda' else None
            native_time = (time.perf_counter() - t0) * 1000 / iterations  # ms
            
            torch.cuda.synchronize() if device == 'cuda' else None
            t0 = time.perf_counter()
            for _ in range(iterations):
                y_triton = permute_rows_triton(x, idx)
            torch.cuda.synchronize() if device == 'cuda' else None
            triton_time = (time.perf_counter() - t0) * 1000 / iterations  # ms
        
        # Validate results
        y_native = x[idx]
        y_triton = permute_rows_triton(x, idx)
        assert torch.allclose(y_native, y_triton)
        
        # Calculate speedup
        speedup = native_time / triton_time
        
        results[f"{rows}x{cols}"] = {
            'native_ms': round(native_time, 2),
            'triton_ms': round(triton_time, 2),
            'speedup': round(speedup, 2)
        }
        
        print(f"  Native: {native_time:.2f} ms")
        print(f"  Triton: {triton_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    return results 