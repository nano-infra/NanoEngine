# NanoDeploy/setup.py

import os
import sys

import pybind11
from setuptools import Extension, find_packages, setup

# 基础编译参数
extra_compile_args = ["-std=c++17", "-O3"]
extra_link_args = []

# OpenMP 配置
# Linux (GCC)
extra_compile_args.append("-fopenmp")
extra_link_args.append("-fopenmp")

ext_modules = [
    Extension(
        "nanodeploy.engine._core",
        sources=["csrc/sequence_binding.cpp", "csrc/scheduler_binding.cpp", "csrc/block_manager_binding.cpp"],
        include_dirs=["csrc", pybind11.get_include(), pybind11.get_include(user=True)],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="nanodeploy",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    zip_safe=False,
)
