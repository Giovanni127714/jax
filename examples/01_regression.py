"""Regression example: train an MLP to predict y = 2*x1 + 3*x2 + noise.

Usage:
    python examples/01_regression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from src.data.generator import RegressionDataGenerator
from src.models.mlp import MLP, model_summary
from src.training.loss import mse_loss
from src.training.trainer import Trainer
from src.utils.config import Config


def main() -> None:
    # Setup
    config = Config(
        hidden_dim=64,
        output_dim=1,
        num_layers=2,
        learning_rate=1e-3,
        num_steps=1000,
        batch_size=32,
        seed=42,
    )

    # Data
    gen = RegressionDataGenerator(config, random_seed=config.seed)
    X, y = gen.generate(1000)
    X_train, y_train, X_val, y_val = gen.train_val_split(X, y)

    # Model
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=config.num_layers)
    model_summary(model, config)

    # Training
    trainer = Trainer(model, config, loss_fn=mse_loss, val_metric_fn=None)
    history = trainer.train(X_train, y_train, X_val, y_val, num_epochs=10)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="Train Loss")
    log_steps = range(config.log_every, config.log_every * len(history["val_loss"]) + 1, config.log_every)
    plt.plot(log_steps, history["val_loss"], label="Val Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("regression_loss.png")
    print("Saved loss plot to regression_loss.png")

    # Checkpoint
    trainer.save_checkpoint(trainer.state, "regression_checkpoint")


if __name__ == "__main__":
    main()
