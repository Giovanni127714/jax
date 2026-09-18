# JAX/Flax MLP Training Framework

A modular, production-ready framework for training MLPs in JAX + Flax,
built as Phase 1.1 of a custom LLM training pipeline.

## Features

- **JAX + Flax:** GPU-accelerated training with `@jax.jit` compilation
- **Modular architecture:** easy to extend towards transformers and language models
- **Type-safe:** full type hints, JAX array annotations
- **Well-tested:** unit tests for all core modules
- **Experiment tracking:** Weights & Biases integration (optional)

## Quick Start (5 minutes)

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/jax-mlp-training.git
cd jax-mlp-training
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Run the regression example
python examples/01_regression.py

# Run the classification example
python examples/02_classification.py
```

## Web UI

The easiest way to use this project is through the local browser dashboard
in [webapp/](webapp/) — no Python code required. It lets you configure and
start a training run, watch loss/accuracy update live, and test the trained
model interactively, all from the browser.

```bash
python webapp/app.py
```

This opens `http://127.0.0.1:5000` automatically. On Windows you can also
just double-click [run_server.bat](run_server.bat).

There are also two minimal CLI helpers for quick checks without the UI:
[play.py](play.py) (`play.bat`) trains a small model and lets you type
values to see predictions in the terminal, and `test.bat` runs the test
suite.

### Deploying the web UI to Vercel

The repo includes [vercel.json](vercel.json), a scoped-down
[webapp/requirements.txt](webapp/requirements.txt), and a `.vercelignore` so
`vercel deploy` (or a GitHub-connected Vercel project) can pick this up as a
Python Function. Two real constraints of that setup, by design:

- **Bundle size.** JAX + Flax pull in `jaxlib`, `scipy`, and `numpy`, which
  together are close to (or over) Vercel's 500 MB standard Function bundle
  limit. If the deploy fails on size, enable **Fluid Compute** / **Large
  Functions** (up to 5 GB) in your Vercel project settings — there is no
  smaller-bundle fallback for this stack.
- **No background thread.** Locally, training runs in a daemon thread so the
  UI can poll for live progress. A serverless request has no such
  guarantee, so on Vercel (detected via the `VERCEL` env var) training runs
  *synchronously inside the request* instead: the page won't show live
  step-by-step updates there, only the final result once the request
  completes. Keep `num_steps` modest so a run finishes within your
  function's `maxDuration`.

This is a best-effort deployment of a project that's architecturally a
better fit for a platform built for long-running processes (a VPS, Render,
Railway, etc.) — treat it as a demo, not a robust production setup.

## Architecture

```
src/
├── models/     Flax model definitions (MLP)
├── training/   Loss functions, metrics, TrainState, and the Trainer class
├── data/       Synthetic data generators (regression & classification)
└── utils/      Config dataclass and shared utilities
webapp/         Local Flask dashboard: train and test the model in a browser
examples/       Standalone, runnable training scripts
tests/          Unit tests for all core modules
configs/        Default hyperparameters and W&B sweep configs
```

## Training Regression

```python
from src.utils.config import Config
from src.data.generator import RegressionDataGenerator
from src.models.mlp import MLP
from src.training.trainer import Trainer
from src.training.loss import mse_loss

config = Config(hidden_dim=64, output_dim=1, num_layers=2, learning_rate=1e-3)
gen = RegressionDataGenerator(config, random_seed=config.seed)
X, y = gen.generate(1000)
X_train, y_train, X_val, y_val = gen.train_val_split(X, y)

model = MLP(hidden_dim=64, output_dim=1, num_layers=2)
trainer = Trainer(model, config, loss_fn=mse_loss)
history = trainer.train(X_train, y_train, X_val, y_val, num_epochs=10)
```

## Training Classification

```python
from src.data.generator import ClassificationDataGenerator
from src.training.loss import cross_entropy_loss, accuracy

config.output_dim = 10
gen = ClassificationDataGenerator(config, num_classes=10)
X, y = gen.generate(2000)
X_train, y_train, X_val, y_val = gen.train_val_split(X, y)

model = MLP(hidden_dim=128, output_dim=10, num_layers=3)
trainer = Trainer(model, config, loss_fn=cross_entropy_loss, val_metric_fn=accuracy)
history = trainer.train(X_train, y_train, X_val, y_val, num_epochs=20)
```

## Configuration

All hyperparameters live in the `Config` dataclass
([src/utils/config.py](src/utils/config.py)). Construct one directly, or
start from a YAML file in [configs/](configs/):

```python
from src.utils.config import Config

config = Config(
    hidden_dim=256,
    num_layers=4,
    learning_rate=5e-4,
    batch_size=64,
    dropout=0.2,
    seed=42,
)
```

## Checkpointing

`Trainer` saves model parameters (via `orbax`), the step count, and the
config to a directory:

```python
# Train, then save
history = trainer.train(X_train, y_train, X_val, y_val, num_epochs=10)
trainer.save_checkpoint(trainer.state, "my_checkpoint")

# Load into a fresh Trainer (optimizer state is reinitialized)
state = trainer.load_checkpoint("my_checkpoint")
```

## Weights & Biases

```bash
wandb login
python examples/03_with_wandb.py
```

If no W&B API key is configured, the script automatically falls back to
offline mode so it still runs end-to-end. Set `config.use_wandb = True`
on any `Trainer` to log training/validation metrics as you go.

Sweep configs for hyperparameter tuning live in
[configs/regression.yaml](configs/regression.yaml) and
[configs/classification.yaml](configs/classification.yaml):

```bash
wandb sweep configs/classification.yaml
wandb agent <sweep_id>
```

## Testing

```bash
pytest tests/ -v
```

## Next Steps

This framework is the foundation for later phases of the custom LLM
training pipeline:

1. **Phase 1.2:** Transformer blocks, attention, positional encoding
2. **Phase 2:** Real data (e.g. a code corpus)
3. **Phase 3:** Full LLM training pipeline

## Contributing

- Format code: `black src/ examples/ tests/ webapp/ play.py`
- Lint: `flake8 src/ examples/ tests/ webapp/ play.py`
- Add tests for new features
- Follow Google-style docstrings

## License

MIT
