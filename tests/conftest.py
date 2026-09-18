"""Shared pytest fixtures for the test suite."""

import pytest

from src.utils.config import Config


@pytest.fixture
def seed() -> int:
    """A fixed seed used across tests for reproducibility."""
    return 42


@pytest.fixture
def small_config() -> Config:
    """A small Config suitable for fast unit tests."""
    return Config(hidden_dim=16, output_dim=4, num_layers=2, input_dim=8)
