"""Hyperparameter configuration for the training framework."""

from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameter configuration.

    Groups all hyperparameters used across data generation, model
    construction, and training into a single, serializable object.

    Attributes:
        hidden_dim: Width of each hidden layer.
        output_dim: Dimensionality of the model output.
        num_layers: Number of hidden layers in the MLP.
        dropout: Dropout probability applied after each hidden layer.
        batch_size: Number of examples per training batch.
        learning_rate: Base learning rate for the optimizer.
        num_steps: Total number of optimizer steps to run.
        seed: Random seed for reproducibility.
        input_dim: Dimensionality of the input features.
        num_samples: Number of samples to generate for synthetic datasets.
        gradient_clip: Maximum absolute value for gradient clipping.
        weight_decay: Weight decay coefficient used by the optimizer.
        log_every: Number of steps between progress log lines.

    Example:
        >>> config = Config(hidden_dim=64, output_dim=1, num_layers=2)
        >>> config.learning_rate
        0.001
    """

    # Model
    hidden_dim: int = 128
    output_dim: int = 10
    num_layers: int = 3
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_steps: int = 1000
    seed: int = 42

    # Data
    input_dim: int = 20
    num_samples: int = 1000

    # Optimization
    gradient_clip: float = 1.0
    weight_decay: float = 0.01

    # Logging
    log_every: int = 100
