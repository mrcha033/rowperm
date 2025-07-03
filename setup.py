import os
import sys
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# Read version from the package
with open(os.path.join('torch_rowperm', '__init__.py'), 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"\'')
            break
    else:
        version = '0.1.0'

# Get the long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

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
    name='torch_rowperm',
    version=version,
    author='rowperm developers',
    author_email='rowperm@example.com',
    description='Fast row permutation operations for PyTorch tensors',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/username/rowperm',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'black>=22.0.0',
            'isort>=5.0.0',
        ],
        'triton': [
            'triton>=2.0.0',
        ],
    },
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    package_data={
        'torch_rowperm': ['_cuda/*.cpp', '_cuda/*.cu', 'py.typed'],
    },
) 