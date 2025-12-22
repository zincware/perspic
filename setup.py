"""
Configuration file for the package.
"""

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="perspic",
    version="0.0.1",
    author="Konstantin Nikolaou and Jonas Scheunemann",
    author_email="jscheunemann@icp.uni-stuttgart.de",
    description="A tool to study neural network training dynamics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=required,
)
