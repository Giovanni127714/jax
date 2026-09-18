"""Tests for src/models/mlp.py"""

import jax
import pytest
import jax.numpy as jnp
from jax import random

from src.models.mlp import MLP, model_summary
from src.utils.config import Config


@pytest.fixture
def config():
    return Config(hidden_dim=64, output_dim=10, input_dim=20)


def test_mlp_init(config):
    """Test MLP initialization."""
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim)
    key = random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, config.input_dim)))
    assert "params" in params


def test_mlp_forward_shape(config):
    """Test MLP forward pass output shape."""
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=2)
    key = random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, config.input_dim)))

    x = jnp.ones((4, config.input_dim))
    output = model.apply(params, x, training=False)

    assert output.shape == (4, config.output_dim)


def test_mlp_dtype(config):
    """Test MLP output dtype."""
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim)
    key = random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, config.input_dim), dtype=jnp.float32))

    x = jnp.ones((1, config.input_dim), dtype=jnp.float32)
    output = model.apply(params, x, training=False)

    assert output.dtype == jnp.float32


def test_mlp_num_layers_affects_param_count(config):
    """More hidden layers should mean more parameters."""
    key = random.PRNGKey(0)
    dummy = jnp.ones((1, config.input_dim))

    shallow = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=1)
    deep = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=3)

    shallow_params = shallow.init(key, dummy)
    deep_params = deep.init(key, dummy)

    shallow_count = sum(p.size for p in jax.tree.leaves(shallow_params))
    deep_count = sum(p.size for p in jax.tree.leaves(deep_params))

    assert deep_count > shallow_count


def test_model_summary(config, capsys):
    """Test model_summary prints correctly."""
    model = MLP(hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_layers=2)

    model_summary(model, config)
    captured = capsys.readouterr()
    assert "Total parameters" in captured.out
