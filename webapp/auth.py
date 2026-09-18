"""Password hashing and session-based login helpers."""

import re
from functools import wraps
from typing import Callable

from flask import jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import db

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
    """Requires a logged-in session, and that the session's user still
    actually exists in the database.

    A stale session (e.g. the database was reset, or the user was deleted)
    would otherwise pass the plain "is there a user_id in the cookie?"
    check and then blow up with a foreign-key error deep in a route -
    confusing for the user, and it's the same handful of extra bytes to
    just verify the session up front and bounce back to login instead.
    """

    @wraps(view)
    def wrapped(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id or db.get_user_by_id(user_id) is None:
            session.clear()
            if request.path.startswith("/api/"):
                return jsonify({"error": "Sessie verlopen. Log opnieuw in."}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped
