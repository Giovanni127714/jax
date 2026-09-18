"""Classification example: train an MLP on a 10-class classification task.

Usage:
    python examples/02_classification.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from src.data.generator import ClassificationDataGenerator
from src.models.mlp import MLP, model_summary
from src.training.loss import accuracy, cross_entropy_loss
from src.training.trainer import Trainer
from src.utils.config import Config


def main() -> None:
    # Setup
    config = Config(
        hidden_dim=128,
        output_dim=10,  # 10 classes
        num_layers=3,
        learning_rate=1e-3,
        num_steps=2000,
        batch_size=32,
        seed=42,
    )

    # Data
    gen = ClassificationDataGenerator(config, num_classes=10, random_seed=config.seed)
    X, y = gen.generate(2000)
    X_train, y_train, X_val, y_val = gen.train_val_split(X, y)

    # Model
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=config.num_layers)
    model_summary(model, config)

    # Training with accuracy metric
    trainer = Trainer(model, config, loss_fn=cross_entropy_loss, val_metric_fn=accuracy)
    history = trainer.train(X_train, y_train, X_val, y_val, num_epochs=20)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    log_steps = range(config.log_every, config.log_every * len(history["val_loss"]) + 1, config.log_every)

    ax1.plot(history["loss"], label="Train Loss")
    ax1.plot(log_steps, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.plot(log_steps, history["val_metric"], label="Val Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("classification_metrics.png")
    print("Saved metrics plot to classification_metrics.png")


if __name__ == "__main__":
    main()
