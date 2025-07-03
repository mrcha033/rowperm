#include <torch/extension.h>

// Forward declarations
torch::Tensor row_permute_cuda(torch::Tensor input, torch::Tensor indices);

// Check if tensor can be processed efficiently without contiguous conversion
bool can_process_directly(const torch::Tensor& tensor) {
    // Check if tensor is already contiguous or if it has a simple strided layout
    // where the first dimension has the largest stride
    if (tensor.is_contiguous()) return true;
    
    auto strides = tensor.strides();
    auto sizes = tensor.sizes();
    
    // For 2D tensors, check if we have a standard row-major layout
    if (tensor.dim() == 2) {
        return strides[0] >= strides[1] * sizes[1];
    }
    
    // For higher dimensions, we need more complex checks
    // This is a simplified check that works for common cases
    int64_t expected_stride = 1;
    for (int i = tensor.dim() - 1; i >= 0; --i) {
        if (strides[i] < expected_stride) return false;
        expected_stride *= sizes[i];
    }
    
    return true;
}

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
    
    // Only make contiguous if necessary
    auto input_proc = can_process_directly(input) ? input : input.contiguous();
    auto indices_proc = indices.is_contiguous() ? indices : indices.contiguous();
    
    return row_permute_cuda(input_proc, indices_proc);
}

// In-place version (for advanced users)
torch::Tensor row_permute_inplace(torch::Tensor input, torch::Tensor indices) {
    // Same validation as above
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");
    TORCH_CHECK(indices.dim() == 1, "Indices must be 1D");
    TORCH_CHECK(indices.size(0) == input.size(0), 
                "Indices length must match input's first dimension");
    TORCH_CHECK(indices.dtype() == torch::kLong, "Indices must be int64");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "Indices must be a CUDA tensor");
    TORCH_CHECK(input.device() == indices.device(), 
                "Input and indices must be on the same device");
    
    // In-place requires contiguous - but we warn instead of forcing
    if (!input.is_contiguous()) {
        TORCH_WARN("Input tensor is not contiguous. In-place operation may not work as expected.");
    }
    
    // For in-place, we create a temporary copy, do the permutation, and copy back
    auto temp = torch::empty_like(input);
    auto indices_proc = indices.is_contiguous() ? indices : indices.contiguous();
    
    // Call CUDA kernel to do the permutation to temp
    auto result = row_permute_cuda(input, indices_proc);
    
    // Copy back to input
    input.copy_(result);
    
    return input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_permute", &row_permute, "Row permutation (CUDA)");
    m.def("row_permute_inplace", &row_permute_inplace, "In-place row permutation (CUDA)");
} 