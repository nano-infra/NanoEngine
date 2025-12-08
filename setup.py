import os
import sys

import pybind11
from setuptools import Extension, find_packages, setup

extra_compile_args = ["-std=c++17", "-O3"]

if sys.platform == "darwin":
    extra_compile_args.append("-stdlib=libc++")
    extra_compile_args.append("-mmacosx-version-min=10.14")

ext_modules = [
    Extension(
        "nanodeploy.engine._core",
        ["csrc/sequence_binding.cpp"],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="nanodeploy",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    zip_safe=False,
)
