# JAX/Flax MLP Training Framework

A modular, production-ready framework for training MLPs in JAX + Flax,
built as Phase 1.1 of a custom LLM training pipeline. The primary way to
use it is the local Claude-powered chat webapp; the underlying `src/`
package is also usable directly from Python.

## Features

- **JAX + Flax:** GPU-accelerated training with `@jax.jit` compilation
- **Modular architecture:** easy to extend towards transformers and language models
- **Type-safe:** full type hints, JAX array annotations
- **Well-tested:** unit tests for all core modules

## Web UI

The main way to use this project is the local browser chat app in
[webapp/](webapp/) — a real Claude-powered assistant that can hold an open
conversation *and* train/test the model when you ask it to, with login and
multi-conversation chat history saved to MySQL.

**Setup (one-time):**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Make sure Laragon's MySQL is running (default: `root` user, no password —
   the app creates its own `jax_chat` database automatically).
2. Copy [.env.example](.env.example) to `.env` in the project root and fill
   in `ANTHROPIC_API_KEY` (get one at [console.anthropic.com](https://console.anthropic.com)).
   Without this, the chat still loads but replies with a clear message
   telling you the key is missing — nothing crashes.

```bash
python webapp/app.py
```

This opens `http://127.0.0.1:5000` automatically. Register an account on
first visit; your conversations persist across logins.

## Architecture

```
src/
├── models/     Flax model definitions (MLP)
├── training/   Loss functions, metrics, TrainState, and the Trainer class
├── data/       Synthetic data generators (regression & classification)
└── utils/      Config dataclass and shared utilities
webapp/         Flask chat app (Claude + tool use, login, MySQL history)
tests/          Unit tests for all core modules
```

## Using src/ directly from Python

The webapp's tools are thin wrappers around this same API, so it's also
usable standalone:

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

For classification, swap in `ClassificationDataGenerator`,
`cross_entropy_loss`, and pass `val_metric_fn=accuracy` to `Trainer`.

## Checkpointing

`Trainer` saves model parameters (via `orbax`), the step count, and the
config to a directory:

```python
history = trainer.train(X_train, y_train, X_val, y_val, num_epochs=10)
trainer.save_checkpoint(trainer.state, "my_checkpoint")

# Load into a fresh Trainer (optimizer state is reinitialized)
state = trainer.load_checkpoint("my_checkpoint")
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

- Format code: `black src/ tests/ webapp/`
- Lint: `flake8 src/ tests/ webapp/`
- Add tests for new features
- Follow Google-style docstrings

## License

MIT
