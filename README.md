# JAX/Flax MLP Training Framework

A modular, production-ready framework for training MLPs in JAX + Flax,
built as Phase 1.1 of a custom LLM training pipeline.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
python examples/01_regression.py
```

This generates synthetic regression data, trains a small MLP, and saves a
loss curve to `regression_loss.png`.

## Architecture

- `src/models/` — Flax model definitions (MLP)
- `src/training/` — Loss functions, metrics, and the `Trainer` class
- `src/data/` — Synthetic data generators for regression/classification
- `src/utils/` — Config system and shared utilities
- `examples/` — Standalone, runnable training scripts
- `tests/` — Unit tests for all core modules

More detailed usage and design notes will be added as the project grows.
