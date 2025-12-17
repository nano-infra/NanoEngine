import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = 'Debug' if debug else 'Release'

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        # Ensure we use a Python-version-compatible pybind11.
        # Otherwise CMake may pick up an old system pybind11 (e.g. /usr/include)
        # which fails to compile with Python 3.12.
        try:
            import pybind11  # type: ignore

            cmake_args += [f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"]
        except Exception:
            pass
        build_args = []
        
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f'-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}']

        is_msvc = False
        if hasattr(self, "compiler") and self.compiler:
            is_msvc = self.compiler.compiler_type == "msvc"
        elif sys.platform.startswith("win32"):
            is_msvc = True

        if not is_msvc:
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all of this mess
            # cl.exe /MP ...
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401
                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", "x64"]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or setuptools
            # by default, so we do this instead.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

setup(
    name="nanodeploy",
    version="0.2.0",
    author="Jinming Ma",
    description="a lightweight LMDeploy implementation built from scratch",
    long_description="",
    ext_modules=[CMakeExtension("nanodeploy._cpp._nanodeploy_cpp", sourcedir="csrc")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "torch>=2.4.0",
        "triton>=3.0.0",
        "transformers>=4.51.0",
        "pydantic",
        "ray",
        "flash-attn",
        "xxhash",
        "dlslime>=0.0.1.post9",
        "dlblas"
    ],
)
