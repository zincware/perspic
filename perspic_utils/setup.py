from setuptools import setup, find_packages

setup(
    name="perspic-utils",
    version="0.1.0",
    description="Utility models and tools for perspic analysis",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "pytorch-lightning>=1.5.0",
        "torchvision>=0.10.0",
    ],
)
