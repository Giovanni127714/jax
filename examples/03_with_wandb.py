"""Classification example with Weights & Biases tracking.

Logs training loss, validation loss, validation accuracy, and the
learning rate to a W&B dashboard. Falls back to offline mode
automatically when no W&B API key is configured, so this script runs
end-to-end even without a W&B account.

Usage:
    wandb login          # optional, enables online dashboard syncing
    python examples/03_with_wandb.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wandb

from src.data.generator import ClassificationDataGenerator
from src.models.mlp import MLP, model_summary
from src.training.loss import accuracy, cross_entropy_loss
from src.training.trainer import Trainer
from src.utils.config import Config


def main() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")

    wandb.init(
        project="jax-mlp-training",
        name="classification-v1",
        config={
            "hidden_dim": 128,
            "output_dim": 10,
            "num_layers": 3,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "num_steps": 2000,
            "seed": 42,
        },
    )

    config = Config(
        hidden_dim=wandb.config.hidden_dim,
        output_dim=wandb.config.output_dim,
        num_layers=wandb.config.num_layers,
        learning_rate=wandb.config.learning_rate,
        batch_size=wandb.config.batch_size,
        num_steps=wandb.config.num_steps,
        seed=wandb.config.seed,
        use_wandb=True,
    )

    # Data
    gen = ClassificationDataGenerator(config, num_classes=config.output_dim, random_seed=config.seed)
    X, y = gen.generate(2000)
    X_train, y_train, X_val, y_val = gen.train_val_split(X, y)

    # Model
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=config.num_layers)
    model_summary(model, config)

    # Training (Trainer logs to W&B internally since config.use_wandb=True)
    trainer = Trainer(model, config, loss_fn=cross_entropy_loss, val_metric_fn=accuracy)
    trainer.train(X_train, y_train, X_val, y_val, num_epochs=20)

    wandb.finish()


if __name__ == "__main__":
    main()
