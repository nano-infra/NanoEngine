from setuptools import find_packages, setup

setup(
    name="nanofold",
    version="0.1.0",
    description="NanoFold structure prediction service (Protenix/AlphaFold3)",
    python_requires=">=3.10",
    packages=find_packages("..", include=["NanoFold", "NanoFold.*"]),
    package_dir={"": ".."},
)
