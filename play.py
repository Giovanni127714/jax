"""Interactive playground: trains a small model, then lets you test it live.

Usage:
    python play.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp

from src.data.generator import RegressionDataGenerator
from src.models.mlp import MLP
from src.training.loss import mse_loss
from src.training.trainer import Trainer
from src.utils.config import Config


def main() -> None:
    print("Model wordt getraind... (een paar seconden)")

    config = Config(
        hidden_dim=64,
        output_dim=1,
        num_layers=2,
        num_steps=300,
        input_dim=20,
        log_every=300,
    )
    gen = RegressionDataGenerator(config, random_seed=42)
    X, y = gen.generate(500)
    X_train, y_train, X_val, y_val = gen.train_val_split(X, y)

    model = MLP(
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_layers=config.num_layers,
    )
    trainer = Trainer(model, config, loss_fn=mse_loss)
    trainer.train(X_train, y_train, X_val, y_val)

    print()
    print("Klaar! Dit model probeert te leren: y = 2*x1 + 3*x2 + ruis")
    print(
        "Typ twee getallen (bv. '1 2') om een voorspelling te zien, of 'stop' om te stoppen."
    )
    print()

    while True:
        raw = input("x1 x2 > ").strip()
        if raw.lower() in ("stop", "quit", "exit", ""):
            break

        parts = raw.split()
        if len(parts) != 2:
            print("Geef twee getallen gescheiden door een spatie, bv: 1 2")
            continue

        try:
            x1, x2 = float(parts[0]), float(parts[1])
        except ValueError:
            print("Dat zijn geen geldige getallen, probeer opnieuw.")
            continue

        x = jnp.zeros((1, config.input_dim)).at[0, 0].set(x1).at[0, 1].set(x2)
        prediction = model.apply({"params": trainer.state.params}, x, training=False)
        true_value = 2 * x1 + 3 * x2

        print(
            f"  AI voorspelt: {float(prediction[0, 0]):.3f}   (echte formule: {true_value:.3f})"
        )

    print("Tot ziens!")


if __name__ == "__main__":
    main()
