"""Flax MLP model definition and inspection utilities."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

from src.utils.config import Config


class MLP(nn.Module):
    """A configurable multi-layer perceptron.

    Alternates ``Dense -> ReLU -> Dropout`` for ``num_layers`` hidden
    layers, followed by a final ``Dense`` output layer with no
    activation and no dropout.

    Attributes:
        hidden_dim: Width of each hidden layer.
        output_dim: Dimensionality of the output.
        num_layers: Number of hidden layers.
        dropout: Dropout probability applied after each hidden layer.

    Example:
        >>> from jax import random
        >>> model = MLP(hidden_dim=128, output_dim=10, num_layers=3)
        >>> key = random.PRNGKey(0)
        >>> params = model.init(key, jnp.ones((1, 20)))
        >>> output = model.apply(params, jnp.ones((1, 20)), training=False)
        >>> output.shape
        (1, 10)
    """

    hidden_dim: int
    output_dim: int
    num_layers: int = 3
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Runs a forward pass through the MLP.

        Args:
            x: Input batch of shape ``(batch_size, input_dim)``.
            training: Whether dropout should be applied stochastically.
                When ``False``, dropout is deterministic (a no-op).

        Returns:
            Output of shape ``(batch_size, output_dim)``.
        """
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        x = nn.Dense(self.output_dim)(x)
        return x


def model_summary(model: MLP, config: Config) -> None:
    """Prints a summary of the model architecture and parameter counts.

    Initializes the model on a dummy batch to compute per-layer and
    total parameter counts, then prints a human-readable summary.

    Args:
        model: An (uninitialized) ``MLP`` instance.
        config: Config providing ``input_dim`` for the dummy input.

    Example:
        >>> model = MLP(hidden_dim=128, output_dim=10, num_layers=2)
        >>> model_summary(model, Config(input_dim=20))
        MLP Model Summary
        ==================
        ...
    """
    key = random.PRNGKey(config.seed)
    dummy_input = jnp.ones((1, config.input_dim))
    params = model.init(key, dummy_input, training=False)["params"]

    print("MLP Model Summary")
    print("==================")

    layer_names = sorted(params.keys(), key=lambda name: int(name.split("_")[-1]))

    total_params = 0
    for layer_idx, layer_name in enumerate(layer_names):
        layer_params = params[layer_name]
        kernel_shape = layer_params["kernel"].shape
        layer_count = sum(p.size for p in jax.tree.leaves(layer_params))
        total_params += layer_count
        print(
            f"Layer {layer_idx}: Dense (input:{kernel_shape[0]} -> "
            f"{kernel_shape[1]}) - {layer_count:,} params"
        )

    print("==================")
    print(f"Total parameters: {total_params:,}")
