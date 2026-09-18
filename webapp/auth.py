"""Password hashing and session-based login helpers."""

import re
from functools import wraps
from typing import Callable

from flask import redirect, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


def is_valid_username(username: str) -> bool:
    return bool(USERNAME_RE.match(username))


def current_user_id():
    return session.get("user_id")


def login_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped
