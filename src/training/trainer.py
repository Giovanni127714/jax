"""Training state, the core JIT-compiled training step, and the Trainer."""

import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from flax.training import train_state
from jax import random

from src.utils.config import Config


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


class Trainer:
    """Manages the training loop, checkpointing, and logging.

    Example:
        >>> trainer = Trainer(model, config, loss_fn=mse_loss)
        >>> history = trainer.train(X_train, y_train, X_val, y_val, num_epochs=10)
        >>> trainer.save_checkpoint(trainer.state, "my_checkpoint")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        val_metric_fn: Optional[
            Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ] = None,
    ) -> None:
        """Initializes the trainer.

        Args:
            model: Flax model to train.
            config: Config object with hyperparameters.
            loss_fn: Loss function, e.g. ``mse_loss`` or ``cross_entropy_loss``.
            val_metric_fn: Optional metric evaluated on argmax'd predictions
                during validation, e.g. ``accuracy``. Only meaningful for
                classification; leave as ``None`` for regression.
        """
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.val_metric_fn = val_metric_fn
        self.history: Dict[str, list] = {"loss": [], "val_loss": [], "val_metric": []}
        self.state: Optional[TrainState] = None

    def create_train_state(self, key: jax.Array, x_dummy: jnp.ndarray) -> TrainState:
        """Initializes model parameters and the optimizer state.

        Uses AdamW with a cosine-decay learning rate schedule over
        ``config.num_steps``.

        Args:
            key: PRNG key used for both parameter init and dropout.
            x_dummy: A representative input batch used only to infer shapes.

        Returns:
            A freshly initialized ``TrainState``.
        """
        init_key, dropout_key = random.split(key)
        params = self.model.init(init_key, x_dummy, training=False)["params"]

        schedule = optax.cosine_decay_schedule(
            init_value=self.config.learning_rate,
            decay_steps=max(self.config.num_steps, 1),
        )
        tx = optax.adamw(learning_rate=schedule, weight_decay=self.config.weight_decay)

        return TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            dropout_rng=dropout_key,
        )

    def train_step(
        self, state: TrainState, x: jnp.ndarray, y: jnp.ndarray
    ) -> Tuple[TrainState, jnp.ndarray]:
        """Runs one training step using this trainer's model and loss_fn.

        Thin wrapper around the module-level, JIT-compiled ``train_step``.

        Args:
            state: Current training state.
            x: Input batch.
            y: Target batch.

        Returns:
            A tuple ``(new_state, loss)``.
        """
        return train_step(
            state, x, y, self.model, self.loss_fn, self.config.gradient_clip
        )

    def evaluate(
        self, state: TrainState, x_val: jnp.ndarray, y_val: jnp.ndarray
    ) -> Dict[str, float]:
        """Evaluates the model on a validation set.

        Args:
            state: Current training state.
            x_val: Validation inputs.
            y_val: Validation targets.

        Returns:
            A dict with ``"val_loss"`` and, if ``val_metric_fn`` was
            provided, ``"val_metric"``.
        """
        predictions = self.model.apply({"params": state.params}, x_val, training=False)
        val_loss = float(self.loss_fn(predictions, y_val))
        metrics: Dict[str, float] = {"val_loss": val_loss}

        if self.val_metric_fn is not None:
            pred_labels = jnp.argmax(predictions, axis=-1)
            metrics["val_metric"] = float(self.val_metric_fn(pred_labels, y_val))

        return metrics

    def train(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_val: jnp.ndarray,
        y_val: jnp.ndarray,
        num_epochs: Optional[int] = None,
        on_log: Optional[Callable[[int, int, float, Dict[str, float]], None]] = None,
    ) -> Dict[str, list]:
        """Runs the full training loop.

        Args:
            X_train: Training inputs of shape ``(num_train, input_dim)``.
            y_train: Training targets.
            X_val: Validation inputs.
            y_val: Validation targets.
            num_epochs: Number of passes over the training set. If ``None``,
                trains for ``config.num_steps`` total optimizer steps
                instead.
            on_log: Optional callback invoked every ``config.log_every``
                steps (and on the final step) as
                ``on_log(step, total_steps, loss, metrics)``. Useful for
                streaming progress to a UI.

        Returns:
            A history dict with ``"loss"``, ``"val_loss"``, and
            ``"val_metric"`` lists recorded every ``config.log_every``
            steps. The final training state is also available as
            ``self.state``.
        """
        key = random.PRNGKey(self.config.seed)
        init_key, shuffle_key = random.split(key)

        num_train = X_train.shape[0]
        batch_size = self.config.batch_size
        steps_per_epoch = max(num_train // batch_size, 1)

        total_steps = (
            num_epochs * steps_per_epoch
            if num_epochs is not None
            else self.config.num_steps
        )

        state = self.create_train_state(init_key, X_train[:1])

        step = 0
        while step < total_steps:
            shuffle_key, perm_key = random.split(shuffle_key)
            perm = random.permutation(perm_key, num_train)
            X_shuffled, y_shuffled = X_train[perm], y_train[perm]

            for batch_start in range(0, steps_per_epoch * batch_size, batch_size):
                if step >= total_steps:
                    break

                batch_x = X_shuffled[batch_start : batch_start + batch_size]
                batch_y = y_shuffled[batch_start : batch_start + batch_size]

                state, loss = self.train_step(state, batch_x, batch_y)
                step += 1
                self.history["loss"].append(float(loss))

                if step % self.config.log_every == 0 or step == total_steps:
                    metrics = self.evaluate(state, X_val, y_val)
                    self.history["val_loss"].append(metrics["val_loss"])
                    self.history["val_metric"].append(metrics.get("val_metric"))

                    message = (
                        f"Step {step}/{total_steps} | loss: {loss:.4f} | "
                        f"val_loss: {metrics['val_loss']:.4f}"
                    )
                    if "val_metric" in metrics:
                        message += f" | val_metric: {metrics['val_metric']:.4f}"
                    print(message)

                    if self.config.use_wandb:
                        self._log_wandb(step, loss, metrics)

                    if on_log is not None:
                        on_log(step, total_steps, float(loss), metrics)

        self.state = state
        return self.history

    def _log_wandb(
        self, step: int, loss: jnp.ndarray, metrics: Dict[str, float]
    ) -> None:
        """Logs metrics to Weights & Biases, if installed and active.

        Args:
            step: Current global training step.
            loss: Training loss for the current step.
            metrics: Validation metrics dict from ``evaluate``.
        """
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is not None:
            wandb.log({"loss": float(loss), **metrics}, step=step)

    def save_checkpoint(self, state: TrainState, path: str) -> None:
        """Saves model parameters, step count, and config to disk.

        Args:
            state: Training state to save.
            path: Directory to save the checkpoint into (created if needed).
        """
        checkpoint_dir = Path(path).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(checkpoint_dir / "params", state.params, force=True)

        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        with open(checkpoint_dir / "step.json", "w") as f:
            json.dump({"step": int(state.step)}, f)

        print(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, path: str) -> TrainState:
        """Loads a checkpoint saved by ``save_checkpoint``.

        Reconstructs a fresh optimizer state (the optimizer itself is not
        checkpointed) and restores parameters and the step counter.

        Args:
            path: Directory previously passed to ``save_checkpoint``.

        Returns:
            A ``TrainState`` with restored ``params`` and ``step``.
        """
        checkpoint_dir = Path(path).resolve()

        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(checkpoint_dir / "params")

        with open(checkpoint_dir / "step.json") as f:
            step = json.load(f)["step"]

        key = random.PRNGKey(self.config.seed)
        dummy_x = jnp.ones((1, self.config.input_dim), dtype=jnp.float32)
        state = self.create_train_state(key, dummy_x)
        state = state.replace(params=params, step=step)

        return state
