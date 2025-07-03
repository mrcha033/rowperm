import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the absolute path to the directory containing setup.py
setup_dir = os.path.dirname(os.path.abspath(__file__))

# Check if CUDA is available
def is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# Build extensions list
ext_modules = []

if is_cuda_available():
    ext_modules.append(
        CUDAExtension(
            name='torch_rowperm._C',
            sources=[
                'torch_rowperm/_cuda/row_perm.cpp',
                'torch_rowperm/_cuda/row_perm.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '--ptxas-options=-v',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_89,code=sm_89',
                    '-gencode=arch=compute_90,code=sm_90',
                    '-gencode=arch=compute_90,code=compute_90',  # PTX for forward compat
                ],
            },
            define_macros=[('TORCH_ROWPERM_VERSION', '"0.1.0"')],
        )
    )

setup(
    name="torch_rowperm",
    version="0.1.0",
    description="Fast row permutation operations for PyTorch tensors",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="mrcha033",
    author_email="mrcha033@gmail.com",
    url="https://github.com/mrcha033/rowperm",
    license="MIT",
    python_requires=">=3.9",
    install_requires=["torch>=2.0.0"],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    packages=find_packages(),
    package_data={
        'torch_rowperm': ['*.so', '*.pyd'],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 