"""Tests for src/training/"""

import jax.numpy as jnp
from jax import random

from src.models.mlp import MLP
from src.training.loss import accuracy, cross_entropy_loss, mse_loss, perplexity
from src.training.trainer import Trainer
from src.utils.config import Config


def test_mse_loss():
    """Test MSE loss computation."""
    pred = jnp.array([[1.0], [2.0], [3.0]])
    target = jnp.array([[1.1], [2.1], [2.9]])
    loss = mse_loss(pred, target)

    assert loss.shape == ()  # Scalar
    assert loss > 0
    assert jnp.isfinite(loss)


def test_cross_entropy_loss():
    """Test cross-entropy loss."""
    logits = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    targets = jnp.array([0, 1])
    loss = cross_entropy_loss(logits, targets)

    assert loss.shape == ()
    assert loss > 0
    assert jnp.isfinite(loss)


def test_accuracy():
    """Test accuracy metric."""
    preds = jnp.array([0, 1, 1, 2])
    targets = jnp.array([0, 1, 0, 2])
    acc = accuracy(preds, targets)

    assert acc == 0.75


def test_perplexity():
    """Test perplexity metric."""
    assert perplexity(jnp.array(0.0)) == 1.0


def test_trainer_init():
    """Test Trainer initialization."""
    config = Config(hidden_dim=64, output_dim=10)
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim)
    trainer = Trainer(model, config, loss_fn=cross_entropy_loss, val_metric_fn=accuracy)

    assert trainer.model is model
    assert trainer.config is config


def test_trainer_create_train_state():
    """Test that create_train_state initializes a usable TrainState."""
    config = Config(hidden_dim=16, output_dim=4, input_dim=8, num_layers=1)
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=config.num_layers)
    trainer = Trainer(model, config, loss_fn=cross_entropy_loss, val_metric_fn=accuracy)

    key = random.PRNGKey(0)
    state = trainer.create_train_state(key, jnp.ones((1, config.input_dim)))

    assert state.step == 0
    assert "Dense_0" in state.params


def test_trainer_train_step_updates_state():
    """A single train_step should change params and return a finite loss."""
    config = Config(hidden_dim=16, output_dim=1, input_dim=8, num_layers=1, gradient_clip=1.0)
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=config.num_layers)
    trainer = Trainer(model, config, loss_fn=mse_loss)

    key = random.PRNGKey(0)
    state = trainer.create_train_state(key, jnp.ones((1, config.input_dim)))

    x = random.normal(random.PRNGKey(1), (4, config.input_dim))
    y = random.normal(random.PRNGKey(2), (4, 1))

    new_state, loss = trainer.train_step(state, x, y)

    assert jnp.isfinite(loss)
    assert new_state.step == state.step + 1


def test_trainer_evaluate_reports_val_metric():
    """evaluate() should include val_metric when val_metric_fn is provided."""
    config = Config(hidden_dim=16, output_dim=4, input_dim=8, num_layers=1)
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=config.num_layers)
    trainer = Trainer(model, config, loss_fn=cross_entropy_loss, val_metric_fn=accuracy)

    key = random.PRNGKey(0)
    state = trainer.create_train_state(key, jnp.ones((1, config.input_dim)))

    x_val = random.normal(random.PRNGKey(1), (10, config.input_dim))
    y_val = random.randint(random.PRNGKey(2), (10,), 0, config.output_dim)

    metrics = trainer.evaluate(state, x_val, y_val)

    assert "val_loss" in metrics
    assert "val_metric" in metrics
    assert 0.0 <= metrics["val_metric"] <= 1.0
