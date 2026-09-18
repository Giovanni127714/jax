"""Package metadata and installation config for jax-mlp-training."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="jax-mlp-training",
    version="0.1.0",
    description=(
        "Production-ready, modular JAX + Flax MLP training framework"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Giovanni van der Weegen",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "numpy",
        "matplotlib",
        "wandb",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
