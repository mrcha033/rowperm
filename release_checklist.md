# Release Checklist for v0.1.0

## Pre-release (D-0)

- [ ] Update version in `torch_rowperm/__init__.py` to `0.1.0`
- [ ] Update version in `setup.py` define_macros to `0.1.0`
- [ ] Run tests locally:
  ```bash
  pytest tests/ -v -m "not gpu"  # CPU tests
  pytest tests/ -v -m gpu         # GPU tests (if available)
  ```

## Build & Test (D-0 + 1h)

- [ ] Build wheel locally on Linux with CUDA:
  ```bash
  python -m pip install build
  python -m build --wheel
  ```
- [ ] Test wheel in fresh environment:
  ```bash
  python -m venv test_env
  source test_env/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install dist/torch_rowperm-0.1.0*.whl
  python -c "import torch_rowperm; print(torch_rowperm.__version__)"
  ```

## Tag & Push (D-0 + 2h)

- [ ] Commit all changes:
  ```bash
  git add -A
  git commit -m "Release v0.1.0"
  ```
- [ ] Create signed tag:
  ```bash
  git tag -s v0.1.0 -m "First release: Linux CUDA wheels"
  git push origin main
  git push origin v0.1.0
  ```

## PyPI Upload (D-0 + 3h)

- [ ] Wait for GitHub Actions to build wheels
- [ ] Download artifacts from GitHub Actions
- [ ] Upload to TestPyPI first:
  ```bash
  python -m twine upload --repository testpypi dist/*
  ```
- [ ] Test from TestPyPI:
  ```bash
  pip install -i https://test.pypi.org/simple/ torch-rowperm
  ```
- [ ] Upload to PyPI:
  ```bash
  python -m twine upload dist/*
  ```

## Post-release (D+1)

- [ ] Verify installation:
  ```bash
  pip install torch-rowperm
  ```
- [ ] Create GitHub Release with notes
- [ ] Announce on:
  - [ ] Twitter/X
  - [ ] Reddit (r/MachineLearning)
  - [ ] Discord/Slack channels

## Issue Triage (D+3)

- [ ] Create labels: `bug/linux`, `build/macos`, `build/windows`
- [ ] Monitor for critical issues
- [ ] Plan v0.1.1 if crash reports > 10

## Hotfix Criteria

Release v0.1.1 if any of:
- [ ] Installation failure rate > 5%
- [ ] Critical bug affecting > 10 users
- [ ] Security issue reported 