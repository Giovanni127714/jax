"""Local web UI for training and testing the MLP framework in a browser.

Usage:
    python webapp/app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os
import sys
import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from flask import Flask, jsonify, render_template, request

from src.data.generator import ClassificationDataGenerator, RegressionDataGenerator
from src.models.mlp import MLP
from src.training.loss import accuracy, cross_entropy_loss, mse_loss
from src.training.trainer import Trainer
from src.utils.config import Config

app = Flask(__name__)

# On Vercel, a background thread is not guaranteed to keep running once the
# HTTP response is sent (each request is a serverless invocation), so a
# training job started there must run synchronously within the request
# instead of in a daemon thread. Locally, the daemon thread lets the UI
# poll for live progress while training continues in the background.
ON_VERCEL = os.environ.get("VERCEL") == "1"


class TrainingJob:
    """Thread-safe holder for the state of the single active training run."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status: str = "idle"  # idle | training | done | error
        self.task: str = "regression"
        self.config: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self.current_step: int = 0
        self.total_steps: int = 0
        self.history: Dict[str, List[float]] = {
            "step": [],
            "loss": [],
            "val_loss": [],
            "val_metric": [],
        }
        self.model: Optional[MLP] = None
        self.trainer: Optional[Trainer] = None

    def reset_for_run(self, task: str, config: Config) -> None:
        with self.lock:
            self.status = "training"
            self.task = task
            self.config = asdict(config)
            self.error = None
            self.current_step = 0
            self.total_steps = config.num_steps
            self.history = {"step": [], "loss": [], "val_loss": [], "val_metric": []}
            self.model = None
            self.trainer = None

    def on_log(
        self, step: int, total_steps: int, loss: float, metrics: Dict[str, float]
    ) -> None:
        with self.lock:
            self.current_step = step
            self.total_steps = total_steps
            self.history["step"].append(step)
            self.history["loss"].append(loss)
            self.history["val_loss"].append(metrics.get("val_loss"))
            self.history["val_metric"].append(metrics.get("val_metric"))

    def mark_done(self, model: MLP, trainer: Trainer) -> None:
        with self.lock:
            self.status = "done"
            self.model = model
            self.trainer = trainer

    def mark_error(self, message: str) -> None:
        with self.lock:
            self.status = "error"
            self.error = message

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "task": self.task,
                "config": self.config,
                "error": self.error,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "history": {k: list(v) for k, v in self.history.items()},
                "has_model": self.model is not None,
            }


job = TrainingJob()


def _run_training(task: str, config: Config) -> None:
    try:
        if task == "classification":
            generator = ClassificationDataGenerator(
                config, num_classes=config.output_dim, random_seed=config.seed
            )
            X, y = generator.generate(config.num_samples)
            loss_fn = cross_entropy_loss
            val_metric_fn = accuracy
        else:
            generator = RegressionDataGenerator(config, random_seed=config.seed)
            X, y = generator.generate(config.num_samples)
            loss_fn = mse_loss
            val_metric_fn = None

        X_train, y_train, X_val, y_val = generator.train_val_split(X, y)

        model = MLP(
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        trainer = Trainer(model, config, loss_fn=loss_fn, val_metric_fn=val_metric_fn)
        trainer.train(X_train, y_train, X_val, y_val, on_log=job.on_log)

        job.mark_done(model, trainer)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        job.mark_error(str(exc))


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/train", methods=["POST"])
def start_training():
    with job.lock:
        if job.status == "training":
            return jsonify({"error": "Er loopt al een training."}), 409

    payload = request.get_json(force=True) or {}
    task = payload.get("task", "regression")
    if task not in ("regression", "classification"):
        return (
            jsonify({"error": "task moet 'regression' of 'classification' zijn."}),
            400,
        )

    try:
        output_dim = 1 if task == "regression" else int(payload.get("num_classes", 4))
        config = Config(
            hidden_dim=int(payload.get("hidden_dim", 64)),
            output_dim=output_dim,
            num_layers=int(payload.get("num_layers", 2)),
            dropout=float(payload.get("dropout", 0.1)),
            batch_size=int(payload.get("batch_size", 32)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            num_steps=int(payload.get("num_steps", 300)),
            seed=int(payload.get("seed", 42)),
            input_dim=int(payload.get("input_dim", 4)),
            num_samples=int(payload.get("num_samples", 1000)),
            gradient_clip=float(payload.get("gradient_clip", 1.0)),
            weight_decay=float(payload.get("weight_decay", 0.01)),
            log_every=max(int(payload.get("log_every", 10)), 1),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Ongeldige configuratie: {exc}"}), 400

    if config.hidden_dim < 1 or config.num_layers < 1 or config.input_dim < 1:
        return (
            jsonify({"error": "hidden_dim, num_layers en input_dim moeten >= 1 zijn."}),
            400,
        )
    if config.batch_size < 1 or config.num_samples < config.batch_size:
        return jsonify({"error": "num_samples moet >= batch_size zijn."}), 400

    job.reset_for_run(task, config)

    if ON_VERCEL:
        # Block the request until training finishes; the UI will see the
        # final "done" state on its first status poll instead of live
        # step-by-step updates. Keep num_steps modest on Vercel so this
        # stays within the function's time limit.
        _run_training(task, config)
    else:
        thread = threading.Thread(
            target=_run_training, args=(task, config), daemon=True
        )
        thread.start()

    return jsonify({"status": "started"})


@app.route("/api/train/status")
def training_status():
    return jsonify(job.snapshot())


@app.route("/api/predict", methods=["POST"])
def predict():
    with job.lock:
        if job.status != "done" or job.trainer is None or job.model is None:
            return jsonify({"error": "Er is nog geen getraind model beschikbaar."}), 400
        model = job.model
        trainer = job.trainer
        task = job.task
        input_dim = job.config["input_dim"]

    payload = request.get_json(force=True) or {}
    features = payload.get("features")
    if not isinstance(features, list) or len(features) != input_dim:
        return (
            jsonify(
                {"error": f"'features' moet een lijst van {input_dim} getallen zijn."}
            ),
            400,
        )

    try:
        x = jnp.array([[float(v) for v in features]], dtype=jnp.float32)
    except (TypeError, ValueError):
        return jsonify({"error": "Alle features moeten getallen zijn."}), 400

    output = model.apply({"params": trainer.state.params}, x, training=False)

    if task == "classification":
        probs = jax.nn.softmax(output, axis=-1)[0]
        predicted_class = int(jnp.argmax(probs))
        return jsonify(
            {
                "task": task,
                "predicted_class": predicted_class,
                "probabilities": [float(p) for p in probs],
            }
        )

    return jsonify({"task": task, "prediction": float(output[0, 0])})


@app.route("/api/model/summary")
def model_summary_api():
    with job.lock:
        if job.model is None or job.trainer is None:
            return jsonify({"error": "Er is nog geen getraind model beschikbaar."}), 400
        params = job.trainer.state.params
        config = dict(job.config)

    layer_names = sorted(params.keys(), key=lambda name: int(name.split("_")[-1]))
    layers = []
    total_params = 0
    for idx, name in enumerate(layer_names):
        kernel_shape = params[name]["kernel"].shape
        count = sum(p.size for p in jax.tree.leaves(params[name]))
        total_params += count
        layers.append(
            {
                "index": idx,
                "input": int(kernel_shape[0]),
                "output": int(kernel_shape[1]),
                "params": int(count),
            }
        )

    return jsonify(
        {"layers": layers, "total_params": int(total_params), "config": config}
    )


if __name__ == "__main__":
    port = 5000
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
