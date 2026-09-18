"""Tests for src/data/"""

import jax.numpy as jnp

from src.data.generator import ClassificationDataGenerator, RegressionDataGenerator
from src.utils.config import Config


def test_regression_generator():
    """Test regression data generation."""
    config = Config(input_dim=20, num_samples=100)
    gen = RegressionDataGenerator(config, random_seed=42)
    X, y = gen.generate(100)

    assert X.shape == (100, 20)
    assert y.shape == (100, 1)
    assert X.dtype == jnp.float32
    assert y.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(X))
    assert jnp.all(jnp.isfinite(y))


def test_regression_train_val_split():
    """Test train/val split."""
    config = Config(input_dim=20, num_samples=100)
    gen = RegressionDataGenerator(config)
    X, y = gen.generate(100)
    X_train, y_train, X_val, y_val = gen.train_val_split(X, y, val_ratio=0.2)

    assert X_train.shape[0] == 80
    assert X_val.shape[0] == 20
    assert y_train.shape[0] == 80
    assert y_val.shape[0] == 20


def test_regression_generator_is_reproducible():
    """Same seed should produce identical data."""
    config = Config(input_dim=20)
    gen_a = RegressionDataGenerator(config, random_seed=42)
    gen_b = RegressionDataGenerator(config, random_seed=42)

    X_a, y_a = gen_a.generate(50)
    X_b, y_b = gen_b.generate(50)

    assert jnp.allclose(X_a, X_b)
    assert jnp.allclose(y_a, y_b)


def test_classification_generator():
    """Test classification data generation."""
    config = Config(input_dim=20, num_samples=100)
    gen = ClassificationDataGenerator(config, num_classes=10, random_seed=42)
    X, y = gen.generate(100)

    assert X.shape == (100, 20)
    assert y.shape == (100,)
    assert jnp.max(y) < 10
    assert jnp.min(y) >= 0


def test_classification_train_val_split():
    """Test classification train/val split."""
    config = Config(input_dim=20)
    gen = ClassificationDataGenerator(config, num_classes=5)
    X, y = gen.generate(100)
    X_train, y_train, X_val, y_val = gen.train_val_split(X, y, val_ratio=0.1)

    assert X_train.shape[0] == 90
    assert X_val.shape[0] == 10
