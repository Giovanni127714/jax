"""Synthetic data generators for regression and classification tasks."""

from typing import Tuple

import jax.numpy as jnp
from jax import random

from src.utils.config import Config


class RegressionDataGenerator:
    """Generates synthetic regression data.

    Targets are a noisy linear combination of the first two input
    features: ``y = 2 * x[:, 0] + 3 * x[:, 1] + noise``.

    Example:
        >>> config = Config(input_dim=20)
        >>> gen = RegressionDataGenerator(config, random_seed=0)
        >>> X, y = gen.generate(100)
        >>> X.shape, y.shape
        ((100, 20), (100, 1))
    """

    def __init__(self, config: Config, random_seed: int = 0) -> None:
        """Initializes the generator.

        Args:
            config: Config object providing ``input_dim``.
            random_seed: Seed used to derive the PRNG key.
        """
        self.config = config
        self.key = random.PRNGKey(random_seed)

    def generate(self, num_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generates a synthetic regression dataset.

        Args:
            num_samples: Number of examples to generate.

        Returns:
            A tuple ``(X, y)`` where ``X`` has shape
            ``(num_samples, input_dim)`` and ``y`` has shape
            ``(num_samples, 1)``, both ``float32``.
        """
        self.key, x_key, noise_key = random.split(self.key, 3)

        X = random.normal(
            x_key, (num_samples, self.config.input_dim), dtype=jnp.float32
        )
        noise = 0.1 * random.normal(noise_key, (num_samples, 1), dtype=jnp.float32)
        y = 2.0 * X[:, 0:1] + 3.0 * X[:, 1:2] + noise
        y = y.astype(jnp.float32)

        assert X.shape == (num_samples, self.config.input_dim)
        assert y.shape == (num_samples, 1)
        assert jnp.all(jnp.isfinite(X)) and jnp.all(jnp.isfinite(y))

        return X, y

    def train_val_split(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        val_ratio: float = 0.1,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Splits data into training and validation sets.

        Args:
            X: Input features of shape ``(num_samples, input_dim)``.
            y: Targets of shape ``(num_samples, 1)``.
            val_ratio: Fraction of samples reserved for validation.

        Returns:
            A tuple ``(X_train, y_train, X_val, y_val)``.
        """
        num_samples = X.shape[0]
        num_val = int(num_samples * val_ratio)
        num_train = num_samples - num_val

        X_train, X_val = X[:num_train], X[num_train:]
        y_train, y_val = y[:num_train], y[num_train:]

        return X_train, y_train, X_val, y_val


class ClassificationDataGenerator:
    """Generates synthetic multi-class classification data.

    Example:
        >>> config = Config(input_dim=20)
        >>> gen = ClassificationDataGenerator(config, num_classes=10, random_seed=0)
        >>> X, y = gen.generate(100)
        >>> X.shape, y.shape
        ((100, 20), (100,))
    """

    def __init__(
        self,
        config: Config,
        num_classes: int = 10,
        random_seed: int = 0,
    ) -> None:
        """Initializes the generator.

        Args:
            config: Config object providing ``input_dim``.
            num_classes: Number of distinct class labels.
            random_seed: Seed used to derive the PRNG key.
        """
        self.config = config
        self.num_classes = num_classes
        self.key = random.PRNGKey(random_seed)

    def generate(self, num_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generates a synthetic classification dataset.

        Class labels are derived from a random linear projection of the
        inputs, followed by an argmax, so that the classes are separable
        but not trivially so.

        Args:
            num_samples: Number of examples to generate.

        Returns:
            A tuple ``(X, y)`` where ``X`` has shape
            ``(num_samples, input_dim)`` (``float32``) and ``y`` has
            shape ``(num_samples,)`` with integer values in
            ``[0, num_classes)``.
        """
        self.key, x_key, proj_key = random.split(self.key, 3)

        X = random.normal(
            x_key, (num_samples, self.config.input_dim), dtype=jnp.float32
        )
        projection = random.normal(
            proj_key, (self.config.input_dim, self.num_classes), dtype=jnp.float32
        )
        logits = X @ projection
        y = jnp.argmax(logits, axis=-1).astype(jnp.int32)

        assert X.shape == (num_samples, self.config.input_dim)
        assert y.shape == (num_samples,)
        assert jnp.all(y >= 0) and jnp.all(y < self.num_classes)

        return X, y

    def train_val_split(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        val_ratio: float = 0.1,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Splits data into training and validation sets.

        Args:
            X: Input features of shape ``(num_samples, input_dim)``.
            y: Labels of shape ``(num_samples,)``.
            val_ratio: Fraction of samples reserved for validation.

        Returns:
            A tuple ``(X_train, y_train, X_val, y_val)``.
        """
        num_samples = X.shape[0]
        num_val = int(num_samples * val_ratio)
        num_train = num_samples - num_val

        X_train, X_val = X[:num_train], X[num_train:]
        y_train, y_val = y[:num_train], y[num_train:]

        return X_train, y_train, X_val, y_val
