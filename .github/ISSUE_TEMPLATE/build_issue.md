---
name: Build Issue
about: Report issues with building from source on macOS/Windows
title: '[BUILD] '
labels: 'build'
assignees: ''

---

**Platform**
- OS: [e.g. macOS 13.1, Windows 11]
- Python version: [e.g. 3.10.8]
- PyTorch version: [e.g. 2.0.1+cu121]
- CUDA version: [e.g. 12.1]

**Build Command**
```bash
# Command you used to install
pip install torch-rowperm --no-binary torch-rowperm
```

**Error Output**
```
# Full error output here
```

**CUDA Setup**
```bash
# Output of nvcc --version
```

**Additional Information**
- How did you install CUDA? [e.g. NVIDIA installer, brew, conda]
- Have you successfully built other PyTorch CUDA extensions? [Yes/No]

**Environment Variables**
```bash
# Any environment variables you set (e.g. CUDA_HOME)
``` 