# torch_rowperm

Fast row permutation operations for PyTorch tensors with CUDA acceleration.

[![PyPI](https://img.shields.io/pypi/v/rowperm)](https://pypi.org/project/rowperm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/mrcha033/rowperm/actions/workflows/test.yml/badge.svg)](https://github.com/mrcha033/rowperm/actions/workflows/test.yml)

## Features

- 🚀 Optimized CUDA kernel for row permutation
- 🔄 Full autograd support
- 📦 Simple `pip install` for Linux users
- 🔧 CPU fallback for non-CUDA tensors
- 🎯 Support for fp32, fp16, and bf16 dtypes
- 🔄 In-place operations for memory efficiency
- ⚡ Optional Triton implementation for even better performance

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- CUDA ≥ 12.1 (for GPU support)
- Triton ≥ 2.0.0 (optional, for maximum performance)

## Installation

### Quick install (Linux + CUDA ≥12.1)

```bash
pip install torch-rowperm
```

### With Triton support (recommended for best performance)

```bash
pip install "torch-rowperm[triton]"
```

### macOS / Windows

Pre-built wheels are not provided yet. Compilation from source is required:

```bash
# macOS
brew install cuda

# Install from source
pip install torch-rowperm --no-binary torch-rowperm
```

**Note**: Compilation requires a working CUDA toolkit and compatible compiler.

## Usage

```python
import torch
import torch_rowperm as rp

# Create a tensor and permutation indices
x = torch.randn(1000, 512, device='cuda')
idx = torch.randperm(1000, device='cuda')

# Standard row permutation
y = rp.permute_rows(x, idx)

# In-place row permutation (memory efficient)
x_clone = x.clone()
rp.permute_rows_(x_clone, idx)  # Note the trailing underscore for in-place

# Supports autograd
x.requires_grad = True
y = rp.permute_rows(x, idx)
y.sum().backward()

# Using Triton implementation (if available)
if rp.HAS_TRITON:
    y_triton = rp.triton_permute_rows(x, idx)
    # In-place version also available
    rp.triton_permute_rows_(x_clone, idx)
```

## Implementation Options

The library provides three implementation paths:

1. **Triton (Default when available)**: Best performance with automatic optimization
   - Uses auto-tuning to select optimal parameters based on tensor size and alignment
   - Specially optimized for various alignment patterns (8-byte and 16-byte aligned data)
   - Requires `torch-rowperm[triton]` installation

2. **CUDA (Default fallback)**: Good performance across all CUDA devices
   - Optimized for memory coalescing and vectorized loads
   - Supports all floating-point types (fp32, fp16, bf16)

3. **Native PyTorch (CPU fallback)**: Uses PyTorch's built-in indexing

The implementation is automatically selected based on availability, with Triton preferred
when installed. You can also explicitly use a specific implementation using the exposed 
API functions.

## Performance

The optimized implementations provide significant speedup over native PyTorch indexing:

| Tensor Size | Native PyTorch | CUDA | Triton | CUDA Speedup | Triton Speedup |
|-------------|----------------|------|--------|--------------|----------------|
| 10K × 512   | 12.5 ms       | 3.9 ms | 2.2 ms | 3.2×         | 5.7×           |
| 100K × 256  | 125 ms        | 31 ms  | 17 ms  | 4.0×         | 7.4×           |

For best performance:
- Use contiguous tensors
- Use tensor sizes that are multiples of 4 elements
- For half/bfloat16, ensure proper alignment
- For maximum performance, use the Triton implementation when available

You can run your own benchmarks with:

```python
import torch_rowperm as rp

# Compare standard implementation with PyTorch
results = rp.benchmark_comparison()
print(results)

# Compare all implementations (if Triton available)
if rp.HAS_TRITON:
    results = rp.benchmark_triton_vs_cuda()
    print(results)
```

## Development

```bash
# Clone the repository
git clone https://github.com/mrcha033/rowperm.git
cd rowperm

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

## Known Issues

- **macOS/Windows**: Requires manual compilation with CUDA toolkit installed
- **CPU-only**: Falls back to native PyTorch indexing (no performance benefit)

## Contributing

Issues and pull requests are welcome! Please include:
- Your PyTorch version: `torch.__version__`
- CUDA version: `nvcc --version`
- Full error logs for build issues

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{torch_rowperm,
  author = {mrcha033},
  title = {torch_rowperm: Fast row permutation for PyTorch},
  year = {2025},
  url = {https://github.com/mrcha033/rowperm}
}
``` 