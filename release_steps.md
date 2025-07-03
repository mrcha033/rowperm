# Step-by-Step Guide to Release torch_rowperm v0.1.0

## 1. Prepare the Environment

```bash
# Create a clean virtual environment
python -m venv release_env
# On Windows
release_env\Scripts\activate
# On Linux/Mac
source release_env/bin/activate

# Install dependencies
pip install wheel setuptools build twine
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 2. Test the Package Locally

```bash
# Run CPU tests
pytest tests/ -v -m "not gpu"

# If you have a CUDA GPU, run GPU tests
pytest tests/ -v -m gpu
```

## 3. Build the Package Locally

```bash
# Windows
build_wheel_local.bat

# Linux/Mac
./build_wheel_local.sh
```

## 4. Commit and Tag

```bash
# Commit all changes
git add -A
git commit -m "Release v0.1.0"

# Create tag (use -s if you have GPG set up)
git tag v0.1.0 -m "First release: Linux CUDA wheels"

# Push to GitHub
git push origin main
git push origin v0.1.0
```

## 5. Monitor GitHub Actions

1. Go to your GitHub repository
2. Click on the "Actions" tab
3. You should see the "Build Wheels" workflow running
4. Wait for it to complete successfully

## 6. Download and Verify Artifacts

1. On the completed workflow run, click on "Summary"
2. Download the "wheels-linux-cuda" and "sdist" artifacts
3. Extract the files to your local `dist/` directory

## 7. Upload to TestPyPI (Optional but Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install -i https://test.pypi.org/simple/ torch-rowperm
```

## 8. Upload to PyPI

```bash
# Upload to real PyPI
python -m twine upload dist/*
```

## 9. Create GitHub Release

1. Go to your GitHub repository
2. Click on "Releases" on the right sidebar
3. Click "Draft a new release"
4. Select the v0.1.0 tag
5. Add a title: "torch_rowperm v0.1.0"
6. Add release notes:
   ```
   First release of torch_rowperm!
   
   ## Features
   - 🚀 Optimized CUDA kernel for row permutation
   - 🔄 Full autograd support
   - 📦 Simple `pip install` for Linux users
   - 🔧 CPU fallback for non-CUDA tensors
   
   ## Installation
   
   ```bash
   pip install torch-rowperm
   ```
   
   ## Known Issues
   - Pre-built wheels available for Linux + CUDA only
   - Mac/Windows users need to compile from source
   ```

7. Attach the wheel files
8. Click "Publish release"

## 10. Verify Installation

```bash
# Create a clean environment
python -m venv test_env
# Activate it
test_env\Scripts\activate  # Windows
source test_env/bin/activate  # Linux/Mac

# Install from PyPI
pip install torch-rowperm

# Verify import works
python -c "import torch_rowperm; print(torch_rowperm.__version__)"
```

## 11. Announce the Release

- Post on Twitter/X
- Share on Reddit r/MachineLearning
- Announce in relevant Discord/Slack channels 