from setuptools import setup

setup(
    name="perspic_utils",
    version="0.1.0",
    description="Utility models and tools for perspic analysis",
    packages=["perspic_utils", "perspic_utils.models"],
    package_dir={"perspic_utils": "."},
    install_requires=[
        "torch>=1.9.0",
        "pytorch-lightning>=1.5.0",
        "torchvision>=0.10.0",
    ],
)
