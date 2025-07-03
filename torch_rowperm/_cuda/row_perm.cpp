#include <torch/extension.h>

// Forward declarations
torch::Tensor row_permute_cuda(torch::Tensor input, torch::Tensor indices);

// C++ interface
torch::Tensor row_permute(torch::Tensor input, torch::Tensor indices) {
    // Input validation
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");
    TORCH_CHECK(indices.dim() == 1, "Indices must be 1D");
    TORCH_CHECK(indices.size(0) == input.size(0), 
                "Indices length must match input's first dimension");
    TORCH_CHECK(indices.dtype() == torch::kLong, "Indices must be int64");
    
    // Device check
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "Indices must be a CUDA tensor");
    TORCH_CHECK(input.device() == indices.device(), 
                "Input and indices must be on the same device");
    
    // Ensure contiguous
    auto input_contig = input.contiguous();
    auto indices_contig = indices.contiguous();
    
    return row_permute_cuda(input_contig, indices_contig);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_permute", &row_permute, "Row permutation (CUDA)");
} 