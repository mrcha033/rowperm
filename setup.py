import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# List of source files for CUDA extension
sources = [
    'torch_rowperm/_cuda/row_perm.cpp'
]

# Determine if CUDA is available
if torch.cuda.is_available() and os.environ.get('FORCE_CPU', '0') != '1':
    print("CUDA is available. Building CUDA extension.")
    sources.append('torch_rowperm/_cuda/row_perm.cu')
    extension = CUDAExtension
    define_macros = [('WITH_CUDA', None)]
    extra_compile_args = {'cxx': ['-O3'], 'nvcc': ['-O3']}
else:
    print("CUDA is NOT available. Building CPU-only extension.")
    extension = CppExtension
    define_macros = []
    extra_compile_args = {'cxx': ['-O3']}

ext_modules = [
    extension(
        name='torch_rowperm._C',
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
) 