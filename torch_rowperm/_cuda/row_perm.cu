#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// Vector types for coalesced memory access
template <typename scalar_t>
struct VectorType { using type = scalar_t; };

template <> struct VectorType<float> { using type = float4; };
template <> struct VectorType<half> { using type = half2; };
template <> struct VectorType<at::BFloat16> { using type = at::BFloat162; };

// Specialization for different data types
template <typename scalar_t, typename vec_t, int VEC_SIZE>
__global__ void row_permute_kernel_vectorized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ indices,
    const int64_t num_rows,
    const int64_t row_size,
    const int64_t vec_elements_per_row) {
    
    // Use shared memory for prefetching
    extern __shared__ char shared_mem[];
    scalar_t* shared_buffer = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Block handles one or more rows
    const int64_t row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row_idx >= num_rows) return;
    
    // Source row to copy from
    const int64_t src_row = indices[row_idx];
    
    // Bounds check
    if (src_row < 0 || src_row >= num_rows) return;
    
    // Source and destination pointers
    const scalar_t* src_ptr = input + src_row * row_size;
    scalar_t* dst_ptr = output + row_idx * row_size;
    
    // Process row in chunks using vectorized loads
    for (int offset = tid; offset < vec_elements_per_row; offset += blockDim.x) {
        if (offset * VEC_SIZE < row_size) {
            // Vector load/store
            vec_t vec_val;
            vec_val = *reinterpret_cast<const vec_t*>(src_ptr + offset * VEC_SIZE);
            *reinterpret_cast<vec_t*>(dst_ptr + offset * VEC_SIZE) = vec_val;
        }
    }
    
    // Handle remaining elements (if row_size is not a multiple of VEC_SIZE)
    const int remainder_start = (row_size / VEC_SIZE) * VEC_SIZE;
    for (int i = remainder_start + tid; i < row_size; i += blockDim.x) {
        dst_ptr[i] = src_ptr[i];
    }
}

// Fallback kernel for small rows or odd sizes
template <typename scalar_t>
__global__ void row_permute_kernel_simple(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ indices,
    const int64_t num_rows,
    const int64_t row_size) {
    
    const int64_t row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row_idx >= num_rows) return;
    
    const int64_t src_row = indices[row_idx];
    
    // Bounds check
    if (src_row < 0 || src_row >= num_rows) return;
    
    // Copy elements with coalesced access pattern
    for (int i = tid; i < row_size; i += blockDim.x) {
        output[row_idx * row_size + i] = input[src_row * row_size + i];
    }
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
    
    // Choose thread block size based on row size
    // For large rows: 1 block per row with many threads
    // For small rows: Multiple blocks with fewer threads
    const int threads_per_block = std::min(1024, std::max(32, (int)row_size / 4));
    const dim3 blocks(num_rows);
    
    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "row_permute_cuda", ([&] {
            
            // Check if row is large enough for vectorization
            if (row_size >= 128 && row_size % 4 == 0) {
                // Use vectorized kernel for float/double
                if (std::is_same<scalar_t, float>::value) {
                    using vec_t = typename VectorType<scalar_t>::type;
                    const int vec_size = sizeof(vec_t) / sizeof(scalar_t);
                    const int vec_elements = row_size / vec_size;
                    
                    row_permute_kernel_vectorized<scalar_t, vec_t, 4><<<blocks, threads_per_block, 0, stream>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        indices.data_ptr<int64_t>(),
                        num_rows,
                        row_size,
                        vec_elements);
                }
                // Use vectorized kernel for half/bfloat16
                else if (std::is_same<scalar_t, at::Half>::value || 
                        std::is_same<scalar_t, at::BFloat16>::value) {
                    using vec_t = typename VectorType<scalar_t>::type;
                    const int vec_size = 2; // half2/bfloat162 contains 2 elements
                    const int vec_elements = row_size / vec_size;
                    
                    row_permute_kernel_vectorized<scalar_t, vec_t, 2><<<blocks, threads_per_block, 0, stream>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        indices.data_ptr<int64_t>(),
                        num_rows,
                        row_size,
                        vec_elements);
                }
                else {
                    // Fallback for other types
                    row_permute_kernel_simple<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        indices.data_ptr<int64_t>(),
                        num_rows,
                        row_size);
                }
            }
            else {
                // Use simple kernel for small rows
                row_permute_kernel_simple<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
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