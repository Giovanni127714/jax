"""Loss functions and metrics for regression and classification."""

import jax.nn
import jax.numpy as jnp


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Computes the mean squared error between predictions and targets.

    Args:
        pred: Predicted values of shape ``(batch_size, ...)``.
        target: Target values with the same shape as ``pred``.

    Returns:
        A scalar MSE loss.

    Example:
        >>> pred = jnp.array([[1.0], [2.0]])
        >>> target = jnp.array([[1.0], [3.0]])
        >>> mse_loss(pred, target)
        Array(0.5, dtype=float32)
    """
    return jnp.mean((pred - target) ** 2)


def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Computes softmax cross-entropy loss for multi-class classification.

    Args:
        logits: Raw model outputs of shape ``(batch_size, num_classes)``.
        targets: Integer class indices of shape ``(batch_size,)``.

    Returns:
        A scalar mean cross-entropy loss.

    Example:
        >>> logits = jnp.array([[2.0, 0.0], [0.0, 2.0]])
        >>> targets = jnp.array([0, 1])
        >>> float(cross_entropy_loss(logits, targets)) < 0.2
        True
    """
    num_classes = logits.shape[-1]
    one_hot_targets = jax.nn.one_hot(targets, num_classes, dtype=jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))


def accuracy(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Computes classification accuracy.

    Args:
        predictions: Predicted integer class indices of shape
            ``(batch_size,)``, typically from ``argmax`` of logits.
        targets: True integer class indices of shape ``(batch_size,)``.

    Returns:
        Scalar accuracy in ``[0, 1]``.

    Example:
        >>> preds = jnp.array([0, 1, 1, 2])
        >>> targets = jnp.array([0, 1, 0, 2])
        >>> accuracy(preds, targets)
        Array(0.75, dtype=float32)
    """
    return jnp.mean((predictions == targets).astype(jnp.float32))


def perplexity(loss: jnp.ndarray) -> jnp.ndarray:
    """Computes perplexity from a cross-entropy loss value.

    Args:
        loss: Scalar cross-entropy loss (natural log base).

    Returns:
        Scalar perplexity, ``exp(loss)``.

    Example:
        >>> perplexity(jnp.array(0.0))
        Array(1.0, dtype=float32)
    """
    return jnp.exp(loss)
