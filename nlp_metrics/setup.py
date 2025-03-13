"""setuptools: A Python package used for packaging and distributing Python projects. It is used here to build and 
install the package, as well as define C extensions.
setup: The main function used to configure and install the package.
Extension: A class used to define a C extension, allowing C code to be included in the Python package.
numpy: The popular numerical computing library. The get_include() function from numpy is used to get the 
path to the NumPy C header files, which may be needed for compiling C extensions that interact with NumPy arrays."""

from setuptools import setup, Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"
"""This part defines a list of C extensions (ext_modules), which will be compiled as part of the setup process.

Extension: This is a setuptools object used to define the C extension. It takes several arguments:
'pycocotools._mask': The name of the extension. It will be compiled as pycocotools/_mask (i.e., the extension will be in the pycocotools package with the name _mask).
sources=['common/maskApi.c', 'pycocotools/_mask.pyx']: These are the source files for the extension:
common/maskApi.c: A C source file, likely containing C code for mask-related operations (COCO dataset is known to use masks for segmentation).
pycocotools/_mask.pyx: This is a Cython file (.pyx extension), which is a Python-like language that compiles to C. It serves as an interface between the Python code and the C code (in maskApi.c).
include_dirs = [np.get_include(), 'common']: This argument specifies the directories to search for header files needed during the compilation:
np.get_include(): This provides the path to NumPy’s header files, which may be needed for working with NumPy arrays in C extensions.
'common': This is likely a directory containing other header files or resources needed for compilation.
extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99']: These are additional arguments to pass to the C compiler:
-Wno-cpp: Suppresses warnings related to the C preprocessor.
-Wno-unused-function: Suppresses warnings for unused functions (useful for reducing noise during compilation).
-std=c99: Specifies that the C code should be compiled according to the C99 standard (a modern version of the C programming language standard).
"""
ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), 'common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir = {'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0'
    ],
    version='2.0',
    ext_modules= ext_modules
)
