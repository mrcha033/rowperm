#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <vector_types.h>

// Proper vector type mappings with correct sizes
template <typename scalar_t>
struct VectorType { using type = scalar_t; };

template <> struct VectorType<float> { using type = float4; };
template <> struct VectorType<at::Half> { using type = __half2; };  // Correct 4-byte size
template <> struct VectorType<at::BFloat16> { using type = __nv_bfloat162; };  // Correct 4-byte size

// Make sure vector types have expected sizes
static_assert(sizeof(float4) == 16, "float4 size mismatch");
static_assert(sizeof(__half2) == 4, "half2 size mismatch");
static_assert(sizeof(__nv_bfloat162) == 4, "bfloat162 size mismatch");

// Constants for kernel configuration
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int ROWS_PER_BLOCK = 1;  // Process one row per block for now
constexpr int ELEMENTS_PER_THREAD = 4;  // Each thread processes multiple elements

// Kernel with shared memory prefetching for improved memory access pattern
template <typename scalar_t, typename vec_t, int VEC_SIZE>
__global__ void row_permute_kernel_vectorized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ indices,
    const int64_t num_rows,
    const int64_t row_size,
    const int64_t vec_elements_per_row) {
    
    // Shared memory for prefetching
    extern __shared__ char shared_mem[];
    scalar_t* shared_buffer = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Calculate row index - handle multi-block processing
    const int64_t row_idx = blockIdx.x / ((row_size + blockDim.x * ELEMENTS_PER_THREAD - 1) / 
                                         (blockDim.x * ELEMENTS_PER_THREAD));
    const int64_t row_block_idx = blockIdx.x % ((row_size + blockDim.x * ELEMENTS_PER_THREAD - 1) / 
                                              (blockDim.x * ELEMENTS_PER_THREAD));
    
    if (row_idx >= num_rows) return;
    
    const int tid = threadIdx.x;
    const int64_t src_row = indices[row_idx];
    
    // Bounds check
    if (src_row < 0 || src_row >= num_rows) return;
    
    // Calculate offsets for this block's portion of the row
    const int64_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const int64_t block_start = row_block_idx * elements_per_block;
    
    // Source and destination pointers
    const scalar_t* src_ptr = input + src_row * row_size + block_start;
    scalar_t* dst_ptr = output + row_idx * row_size + block_start;
    
    // Actual elements to process (handle edge case)
    const int64_t elements_to_process = min(elements_per_block, row_size - block_start);
    const int64_t vec_elements = (elements_to_process + VEC_SIZE - 1) / VEC_SIZE;
    
    // First, collaboratively load data into shared memory with vectorized loads
    // This improves memory coalescing and creates a prefetch stage
    for (int offset = tid; offset < (vec_elements + blockDim.x - 1) / blockDim.x * blockDim.x; offset += blockDim.x) {
        if (offset < vec_elements && (block_start + offset * VEC_SIZE) < row_size) {
            // Make sure we're not reading past the end of the row
            const int64_t src_offset = min(offset * VEC_SIZE, row_size - block_start - VEC_SIZE);
            
            // Ensure alignment for vector loads using __builtin_assume_aligned
            const auto* src_aligned = reinterpret_cast<const vec_t*>(__builtin_assume_aligned(
                src_ptr + src_offset, alignof(vec_t)));
            
            // Load into shared memory
            vec_t vec_val = *src_aligned;
            *reinterpret_cast<vec_t*>(__builtin_assume_aligned(
                shared_buffer + src_offset, alignof(vec_t))) = vec_val;
        }
    }
    
    // Make sure all data is loaded before proceeding
    __syncthreads();
    
    // Now copy from shared memory to output with better memory access pattern
    for (int offset = tid; offset < (elements_to_process + blockDim.x - 1) / blockDim.x * blockDim.x; 
         offset += blockDim.x) {
        if (offset < elements_to_process) {
            if ((block_start + offset + VEC_SIZE) <= row_size && offset % VEC_SIZE == 0) {
                // Vector copy if we have a full vector and proper alignment
                const auto* src_aligned = reinterpret_cast<const vec_t*>(__builtin_assume_aligned(
                    shared_buffer + offset, alignof(vec_t)));
                
                auto* dst_aligned = reinterpret_cast<vec_t*>(__builtin_assume_aligned(
                    dst_ptr + offset, alignof(vec_t)));
                
                *dst_aligned = *src_aligned;
            }
            else {
                // Scalar copy for edge cases and misaligned data
                dst_ptr[offset] = shared_buffer[offset];
            }
        }
    }
}

// Improved scalar fallback kernel
template <typename scalar_t>
__global__ void row_permute_kernel_simple(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ indices,
    const int64_t num_rows,
    const int64_t row_size) {
    
    // Handle multi-block processing for large rows
    const int64_t row_idx = blockIdx.x / ((row_size + blockDim.x * ELEMENTS_PER_THREAD - 1) / 
                                         (blockDim.x * ELEMENTS_PER_THREAD));
    const int64_t row_block_idx = blockIdx.x % ((row_size + blockDim.x * ELEMENTS_PER_THREAD - 1) / 
                                              (blockDim.x * ELEMENTS_PER_THREAD));
    
    if (row_idx >= num_rows) return;
    
    const int tid = threadIdx.x;
    const int64_t src_row = indices[row_idx];
    
    // Bounds check
    if (src_row < 0 || src_row >= num_rows) return;
    
    // Calculate offsets for this block's portion of the row
    const int64_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const int64_t block_start = row_block_idx * elements_per_block;
    
    // Actual elements to process (handle edge case)
    const int64_t elements_to_process = min(elements_per_block, row_size - block_start);
    
    // Use shared memory as a collaborative cache
    extern __shared__ char shared_mem[];
    scalar_t* shared_buffer = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Load data into shared memory
    for (int i = tid; i < elements_to_process; i += blockDim.x) {
        shared_buffer[i] = input[src_row * row_size + block_start + i];
    }
    
    __syncthreads();
    
    // Copy from shared memory to output
    for (int i = tid; i < elements_to_process; i += blockDim.x) {
        output[row_idx * row_size + block_start + i] = shared_buffer[i];
    }
}

// Check if tensor memory layout is suitable for vectorized load/store
bool can_use_vectorized(const torch::Tensor& tensor, int vec_size) {
    // Check if contiguous
    if (tensor.is_contiguous()) {
        // Check alignment of first element (assuming row_size is large enough)
        uintptr_t ptr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        return (ptr % vec_size) == 0;
    }
    return false;
}

torch::Tensor row_permute_cuda(
    torch::Tensor input,
    torch::Tensor indices) {
    
    // Setup CUDA device guard and get current stream
    at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const auto num_rows = input.size(0);
    const auto row_size = input.numel() / num_rows;
    
    auto output = torch::empty_like(input);
    
    // Better thread block sizing based on row size
    const int threads_per_block = std::min(MAX_THREADS_PER_BLOCK, 
                                          std::max(WARP_SIZE, 
                                                  (int)(row_size / (ELEMENTS_PER_THREAD * 2))));
    
    // Calculate number of blocks needed per row and total blocks
    const int blocks_per_row = (row_size + threads_per_block * ELEMENTS_PER_THREAD - 1) / 
                              (threads_per_block * ELEMENTS_PER_THREAD);
    
    // Limit grid size to avoid exceeding CUDA limits (2^31-1)
    const int64_t total_blocks = std::min(
        static_cast<int64_t>(blocks_per_row) * num_rows,
        static_cast<int64_t>(2147483647)  // 2^31-1
    );
    
    // Calculate shared memory size for one row's worth of data we're processing in a block
    const int elements_per_block = threads_per_block * ELEMENTS_PER_THREAD;
    const int shared_mem_size = std::min(static_cast<int>(row_size), elements_per_block) * sizeof(float);
    
    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "row_permute_cuda", ([&] {
            
            // Determine if we can use vectorized loads based on alignment and size
            bool use_vectorized = can_use_vectorized(input, 16) &&  // For float4/half8 alignment
                                 can_use_vectorized(output, 16) &&
                                 row_size >= 128;
            
            if (use_vectorized) {
                if (std::is_same<scalar_t, float>::value) {
                    using vec_t = typename VectorType<scalar_t>::type;
                    const int vec_size = sizeof(vec_t) / sizeof(scalar_t);
                    const int vec_elements = (row_size + vec_size - 1) / vec_size;
                    
                    row_permute_kernel_vectorized<scalar_t, vec_t, 4>
                        <<<total_blocks, threads_per_block, shared_mem_size, stream>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        indices.data_ptr<int64_t>(),
                        num_rows,
                        row_size,
                        vec_elements);
                }
                else if (std::is_same<scalar_t, at::Half>::value) {
                    using vec_t = typename VectorType<scalar_t>::type;
                    const int vec_size = 2;  // half2 contains 2 elements
                    const int vec_elements = (row_size + vec_size - 1) / vec_size;
                    
                    row_permute_kernel_vectorized<scalar_t, vec_t, 2>
                        <<<total_blocks, threads_per_block, shared_mem_size, stream>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        indices.data_ptr<int64_t>(),
                        num_rows,
                        row_size,
                        vec_elements);
                }
                else if (std::is_same<scalar_t, at::BFloat16>::value) {
                    using vec_t = typename VectorType<scalar_t>::type;
                    const int vec_size = 2;  // bfloat162 contains 2 elements
                    const int vec_elements = (row_size + vec_size - 1) / vec_size;
                    
                    row_permute_kernel_vectorized<scalar_t, vec_t, 2>
                        <<<total_blocks, threads_per_block, shared_mem_size, stream>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        indices.data_ptr<int64_t>(),
                        num_rows,
                        row_size,
                        vec_elements);
                }
                else {
                    row_permute_kernel_simple<scalar_t>
                        <<<total_blocks, threads_per_block, shared_mem_size, stream>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        indices.data_ptr<int64_t>(),
                        num_rows,
                        row_size);
                }
            }
            else {
                // Use simple kernel for small rows or when alignment is not suitable
                row_permute_kernel_simple<scalar_t>
                    <<<total_blocks, threads_per_block, shared_mem_size, stream>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(),
                    num_rows,
                    row_size);
            }
        }));
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
} 