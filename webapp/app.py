"""Local web UI for training and testing the MLP framework in a browser.

Usage:
    python webapp/app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import secrets
import sys
import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency): KEY=VALUE per line."""
    if not path.exists():
        return
    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")

import jax
import jax.numpy as jnp
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

import assistant
import auth
import db
from src.data.generator import ClassificationDataGenerator, RegressionDataGenerator
from src.models.mlp import MLP
from src.training.loss import accuracy, cross_entropy_loss, mse_loss
from src.training.trainer import Trainer
from src.utils.config import Config

app = Flask(__name__)


def _get_or_create_secret_key() -> str:
    secret_path = Path(__file__).resolve().parent / ".secret_key"
    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    secret_path.write_text(key, encoding="utf-8")
    return key


app.secret_key = _get_or_create_secret_key()
db.init_db()


@app.context_processor
def inject_asset_version() -> Dict[str, Any]:
    """Exposes asset_version() to templates for cache-busting static files.

    Appending ?v=<mtime> to a static URL forces browsers to refetch it as
    soon as the file changes on disk, instead of serving a stale cached
    copy after a normal reload.
    """

    def asset_version(rel_path: str) -> int:
        full_path = Path(app.static_folder) / rel_path
        try:
            return int(full_path.stat().st_mtime)
        except OSError:
            return 0

    return {"asset_version": asset_version}


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


# ---------------------------------------------------------------------
# Tool implementations (shared by the direct REST routes and the Claude
# tool-calling loop in /api/chat)
# ---------------------------------------------------------------------


def _tool_start_training(tool_input: Dict[str, Any]):
    with job.lock:
        if job.status == "training":
            return {"error": "Er loopt al een training."}, None

    task = tool_input.get("task", "regression")
    if task not in ("regression", "classification"):
        return {"error": "task moet 'regression' of 'classification' zijn."}, None

    try:
        output_dim = (
            1 if task == "regression" else int(tool_input.get("num_classes", 4))
        )
        num_steps = int(tool_input.get("num_steps", 300))
        config = Config(
            hidden_dim=int(tool_input.get("hidden_dim", 64)),
            output_dim=output_dim,
            num_layers=int(tool_input.get("num_layers", 2)),
            dropout=float(tool_input.get("dropout", 0.1)),
            batch_size=int(tool_input.get("batch_size", 32)),
            learning_rate=float(tool_input.get("learning_rate", 1e-3)),
            num_steps=num_steps,
            seed=int(tool_input.get("seed", 42)),
            input_dim=int(tool_input.get("input_dim", 4)),
            num_samples=int(tool_input.get("num_samples", 1000)),
            gradient_clip=float(tool_input.get("gradient_clip", 1.0)),
            weight_decay=float(tool_input.get("weight_decay", 0.01)),
            log_every=max(int(tool_input.get("log_every", max(num_steps // 10, 1))), 1),
        )
    except (TypeError, ValueError) as exc:
        return {"error": f"Ongeldige configuratie: {exc}"}, None

    if config.hidden_dim < 1 or config.num_layers < 1 or config.input_dim < 1:
        return {"error": "hidden_dim, num_layers en input_dim moeten >= 1 zijn."}, None
    if config.batch_size < 1 or config.num_samples < config.batch_size:
        return {"error": "num_samples moet >= batch_size zijn."}, None

    job.reset_for_run(task, config)
    thread = threading.Thread(target=_run_training, args=(task, config), daemon=True)
    thread.start()

    result = {"status": "started", "task": task, "num_steps": config.num_steps}
    action = {"type": "start_training", "num_steps": config.num_steps}
    return result, action


def _tool_predict(tool_input: Dict[str, Any]):
    with job.lock:
        if job.status != "done" or job.trainer is None or job.model is None:
            return {"error": "Er is nog geen getraind model beschikbaar."}, None
        model = job.model
        trainer = job.trainer
        task = job.task
        input_dim = job.config["input_dim"]

    features = tool_input.get("features")
    if not isinstance(features, list) or len(features) != input_dim:
        return {
            "error": f"'features' moet een lijst van {input_dim} getallen zijn."
        }, None

    try:
        x = jnp.array([[float(v) for v in features]], dtype=jnp.float32)
    except (TypeError, ValueError):
        return {"error": "Alle features moeten getallen zijn."}, None

    output = model.apply({"params": trainer.state.params}, x, training=False)

    if task == "classification":
        probs = jax.nn.softmax(output, axis=-1)[0]
        predicted_class = int(jnp.argmax(probs))
        result = {
            "task": task,
            "features": features,
            "predicted_class": predicted_class,
            "probabilities": [float(p) for p in probs],
        }
    else:
        result = {"task": task, "features": features, "prediction": float(output[0, 0])}

    return result, {"type": "predict", "result": result}


def _tool_get_model_summary(_tool_input: Optional[Dict[str, Any]] = None):
    with job.lock:
        if job.model is None or job.trainer is None:
            return {"error": "Er is nog geen getraind model beschikbaar."}, None
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

    result = {"layers": layers, "total_params": int(total_params), "config": config}
    return result, {"type": "model_summary", "result": result}


def _tool_get_training_status(_tool_input: Optional[Dict[str, Any]] = None):
    snap = job.snapshot()
    result = {
        "status": snap["status"],
        "current_step": snap["current_step"],
        "total_steps": snap["total_steps"],
        "error": snap["error"],
    }
    return result, None


def _execute_tool(name: str, tool_input: Dict[str, Any]):
    handlers = {
        "start_training": _tool_start_training,
        "predict": _tool_predict,
        "get_model_summary": _tool_get_model_summary,
        "get_training_status": _tool_get_training_status,
    }
    handler = handlers.get(name)
    if handler is None:
        return {"error": f"Onbekende tool: {name}"}, None
    return handler(tool_input)


# ---------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", error=None)

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    password_confirm = request.form.get("password_confirm") or ""

    if not auth.is_valid_username(username):
        return render_template(
            "register.html",
            error="Gebruikersnaam moet 3-32 tekens zijn (letters, cijfers, _ . -).",
        )
    if len(password) < 8:
        return render_template(
            "register.html", error="Wachtwoord moet minstens 8 tekens lang zijn."
        )
    if password != password_confirm:
        return render_template(
            "register.html", error="Wachtwoorden komen niet overeen."
        )
    if db.get_user_by_username(username):
        return render_template("register.html", error="Deze gebruikersnaam bestaat al.")

    user_id = db.create_user(username, auth.hash_password(password))
    session["user_id"] = user_id
    session["username"] = username
    return redirect(url_for("index"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    user = db.get_user_by_username(username)
    if not user or not auth.verify_password(user["password_hash"], password):
        return render_template(
            "login.html", error="Onjuiste gebruikersnaam of wachtwoord."
        )

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------
# App routes
# ---------------------------------------------------------------------


@app.route("/")
@auth.login_required
def index() -> str:
    return render_template("index.html", username=session.get("username"))


@app.route("/api/history")
@auth.login_required
def history():
    return jsonify({"messages": db.get_messages(auth.current_user_id())})


@app.route("/api/chat", methods=["POST"])
@auth.login_required
def chat():
    user_id = auth.current_user_id()
    payload = request.get_json(force=True) or {}
    user_message = (payload.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Leeg bericht."}), 400

    history_rows = db.get_messages(user_id, limit=40)
    conversation = [{"role": r["role"], "content": r["content"]} for r in history_rows]

    db.save_message(user_id, "user", user_message)

    try:
        reply, action = assistant.chat_turn(conversation, user_message, _execute_tool)
    except assistant.AssistantError as exc:
        reply = str(exc)
        action = None

    db.save_message(user_id, "assistant", reply, attachment=action)

    return jsonify({"reply": reply, "action": action})


@app.route("/api/train", methods=["POST"])
@auth.login_required
def start_training():
    payload = request.get_json(force=True) or {}
    result, _ = _tool_start_training(payload)
    if "error" in result:
        status_code = 409 if "loopt al" in result["error"] else 400
        return jsonify(result), status_code
    return jsonify({"status": "started"})


@app.route("/api/train/status")
@auth.login_required
def training_status():
    return jsonify(job.snapshot())


@app.route("/api/predict", methods=["POST"])
@auth.login_required
def predict():
    payload = request.get_json(force=True) or {}
    result, _ = _tool_predict(payload)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/model/summary")
@auth.login_required
def model_summary_api():
    result, _ = _tool_get_model_summary()
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    port = 5000
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
