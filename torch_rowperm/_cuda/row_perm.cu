#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void row_permute_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ indices,
    const int64_t num_rows,
    const int64_t row_size) {
    
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = num_rows * row_size;
    
    if (idx < total_elements) {
        const int64_t row_idx = idx / row_size;
        const int64_t col_idx = idx % row_size;
        const int64_t src_row = indices[row_idx];
        
        // Bounds check
        if (src_row >= 0 && src_row < num_rows) {
            output[idx] = input[src_row * row_size + col_idx];
        }
    }
}

torch::Tensor row_permute_cuda(
    torch::Tensor input,
    torch::Tensor indices) {
    
    const auto num_rows = input.size(0);
    const auto row_size = input.numel() / num_rows;
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "row_permute_cuda", ([&] {
            row_permute_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                num_rows,
                row_size);
        }));
    
    return output;
} 