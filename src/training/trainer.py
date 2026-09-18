"""Training state and the core JIT-compiled training step."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state


class TrainState(train_state.TrainState):
    """Extends Flax's TrainState with an RNG stream for dropout.

    Attributes:
        dropout_rng: PRNG key consumed (and re-split) on every training
            step to drive stochastic dropout.
    """

    dropout_rng: jax.Array


@partial(jax.jit, static_argnames=("model", "loss_fn"))
def train_step(
    state: TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    model: nn.Module,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    gradient_clip: float,
) -> Tuple[TrainState, jnp.ndarray]:
    """Runs a single JIT-compiled training step.

    Args:
        state: Current training state (params, opt_state, step, dropout_rng).
        x: Input batch of shape ``(batch_size, input_dim)``.
        y: Target batch (shape depends on ``loss_fn``).
        model: Flax model; treated as a static (hashable) argument.
        loss_fn: Loss function ``(predictions, targets) -> scalar``; static.
        gradient_clip: Maximum absolute gradient value (element-wise clip).

    Returns:
        A tuple ``(new_state, loss)`` with the updated training state and
        the scalar loss value for this batch.

    Example:
        >>> state, loss = train_step(
        ...     state, x_batch, y_batch, model, cross_entropy_loss, 1.0
        ... )
    """
    new_dropout_rng, step_rng = jax.random.split(state.dropout_rng)

    def compute_loss(params: dict) -> jnp.ndarray:
        predictions = model.apply(
            {"params": params},
            x,
            training=True,
            rngs={"dropout": step_rng},
        )
        return loss_fn(predictions, y)

    loss, grads = jax.value_and_grad(compute_loss)(state.params)
    grads = jax.tree.map(lambda g: jnp.clip(g, -gradient_clip, gradient_clip), grads)

    state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    return state, loss
