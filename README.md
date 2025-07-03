# torch_rowperm

Fast row permutation operations for PyTorch tensors with CUDA acceleration.

[![PyPI](https://img.shields.io/pypi/v/torch-rowperm)](https://pypi.org/project/torch-rowperm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yunmin/torch_rowperm/actions/workflows/test.yml/badge.svg)](https://github.com/yunmin/torch_rowperm/actions/workflows/test.yml)

## Installation

### Quick install (Linux + CUDA ≥12.1)

```bash
pip install torch-rowperm
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

# Permute rows
y = rp.permute_rows(x, idx)

# Supports autograd
x.requires_grad = True
y = rp.permute_rows(x, idx)
y.sum().backward()
```

## Features

- 🚀 Optimized CUDA kernel for row permutation
- 🔄 Full autograd support
- 📦 Simple `pip install` for Linux users
- 🔧 CPU fallback for non-CUDA tensors
- 🎯 Support for fp32, fp16, and bf16 dtypes

## Performance

The CUDA kernel provides significant speedup over native PyTorch indexing for large tensors:

| Tensor Size | Native PyTorch | torch_rowperm | Speedup |
|-------------|----------------|---------------|---------|
| 10K × 512   | 12.5 ms       | 3.2 ms        | 3.9×    |
| 100K × 256  | 125 ms        | 31 ms         | 4.0×    |

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- CUDA ≥ 12.1 (for GPU support)

## Development

```bash
# Clone the repository
git clone https://github.com/yunmin/torch_rowperm.git
cd torch_rowperm

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
  author = {Yunmin},
  title = {torch_rowperm: Fast row permutation for PyTorch},
  year = {2024},
  url = {https://github.com/yunmin/torch_rowperm}
}
``` 