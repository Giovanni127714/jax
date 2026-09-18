"""MySQL persistence: users, conversations, and chat message history.

Connects to the MySQL server bundled with Laragon (root / no password by
default). Override via DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME env vars
if your setup differs.

Opening a fresh TCP+auth connection per query costs ~30ms even to a local
server. Route handlers reuse one connection for the whole request (via
Flask's request-scoped `g`), closed automatically in app.py's
teardown_appcontext hook; init_db() runs before any request exists, so it
manages its own short-lived connections instead.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pymysql
from flask import g
from pymysql.cursors import DictCursor

DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "jax_chat")

DEFAULT_CONVERSATION_TITLE = "Nieuw gesprek"


def _connect(with_db: bool = True) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME if with_db else None,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )


def _conn() -> pymysql.connections.Connection:
    """Request-scoped connection: one per Flask request, reused across
    every db.py call within it instead of opening a fresh one each time."""
    if "db_conn" not in g:
        g.db_conn = _connect()
    return g.db_conn


def close_request_connection(_exception: Optional[BaseException] = None) -> None:
    """Registered as app.teardown_appcontext; closes the request's connection."""
    conn = g.pop("db_conn", None)
    if conn is not None:
        conn.close()


def init_db() -> None:
    """Creates the database/tables if needed, and migrates older schemas.

    Runs once at startup, outside any request context, so it manages its
    own connections rather than using the request-scoped `_conn()`.

    Safe to call on every app startup: every statement is idempotent, and
    the messages->conversations migration only touches rows that don't
    have a conversation_id yet.
    """
    conn = _connect(with_db=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        conn.close()

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(64) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    title VARCHAR(120) NOT NULL DEFAULT 'Nieuw gesprek',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_user_updated (user_id, updated_at),
                    CONSTRAINT fk_conversations_user FOREIGN KEY (user_id)
                        REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    conversation_id INT NULL,
                    role ENUM('user', 'assistant') NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    attachment JSON NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user_created (user_id, created_at),
                    INDEX idx_conversation_created (conversation_id, created_at),
                    CONSTRAINT fk_messages_user FOREIGN KEY (user_id)
                        REFERENCES users(id) ON DELETE CASCADE,
                    CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

            # Migration: older installs may have a messages table without
            # conversation_id (schema predates multi-conversation support).
            cur.execute(
                """
                SELECT COUNT(*) AS n FROM information_schema.columns
                WHERE table_schema = %s AND table_name = 'messages'
                    AND column_name = 'conversation_id'
                """,
                (DB_NAME,),
            )
            if cur.fetchone()["n"] == 0:
                cur.execute(
                    "ALTER TABLE messages ADD COLUMN conversation_id INT NULL AFTER user_id"
                )
                cur.execute(
                    "ALTER TABLE messages ADD CONSTRAINT fk_messages_conversation "
                    "FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE"
                )

            _backfill_conversations(cur)
    finally:
        conn.close()


def _backfill_conversations(cur) -> None:
    """Wraps any pre-existing user_id-only messages in a single conversation each."""
    cur.execute("SELECT DISTINCT user_id FROM messages WHERE conversation_id IS NULL")
    orphan_user_ids = [row["user_id"] for row in cur.fetchall()]
    for user_id in orphan_user_ids:
        cur.execute(
            "INSERT INTO conversations (user_id, title) VALUES (%s, %s)",
            (user_id, "Eerder gesprek"),
        )
        conversation_id = cur.lastrowid
        cur.execute(
            "UPDATE messages SET conversation_id = %s "
            "WHERE user_id = %s AND conversation_id IS NULL",
            (conversation_id, user_id),
        )


# ---------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------


def create_user(username: str, password_hash: str) -> int:
    with _conn().cursor() as cur:
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, password_hash),
        )
        return cur.lastrowid


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with _conn().cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        return cur.fetchone()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with _conn().cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        return cur.fetchone()


# ---------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------


def create_conversation(user_id: int, title: str = DEFAULT_CONVERSATION_TITLE) -> int:
    with _conn().cursor() as cur:
        cur.execute(
            "INSERT INTO conversations (user_id, title) VALUES (%s, %s)",
            (user_id, title),
        )
        return cur.lastrowid


def list_conversations(user_id: int) -> List[Dict[str, Any]]:
    with _conn().cursor() as cur:
        cur.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = cur.fetchall()
    for row in rows:
        row["created_at"] = row["created_at"].isoformat()
        row["updated_at"] = row["updated_at"].isoformat()
    return rows


def get_conversation(conversation_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    with _conn().cursor() as cur:
        cur.execute(
            "SELECT * FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        )
        return cur.fetchone()


def rename_conversation(conversation_id: int, title: str) -> None:
    with _conn().cursor() as cur:
        cur.execute(
            "UPDATE conversations SET title = %s WHERE id = %s",
            (title[:120], conversation_id),
        )


def touch_conversation(conversation_id: int, title: Optional[str] = None) -> None:
    """Bumps updated_at (for sidebar ordering); optionally renames in the
    same query, so a fresh conversation's first turn only costs one
    round-trip instead of two."""
    with _conn().cursor() as cur:
        if title is not None:
            cur.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP, title = %s "
                "WHERE id = %s",
                (title[:120], conversation_id),
            )
        else:
            cur.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (conversation_id,),
            )


def delete_conversation(conversation_id: int, user_id: int) -> bool:
    with _conn().cursor() as cur:
        cur.execute(
            "DELETE FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------


def save_message(
    user_id: int,
    conversation_id: int,
    role: str,
    content: str,
    attachment: Optional[Dict[str, Any]] = None,
) -> int:
    with _conn().cursor() as cur:
        cur.execute(
            "INSERT INTO messages (user_id, conversation_id, role, content, attachment) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                user_id,
                conversation_id,
                role,
                content,
                json.dumps(attachment) if attachment else None,
            ),
        )
        return cur.lastrowid


def get_messages(conversation_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    with _conn().cursor() as cur:
        cur.execute(
            "SELECT role, content, attachment, created_at FROM messages "
            "WHERE conversation_id = %s ORDER BY id ASC LIMIT %s",
            (conversation_id, limit),
        )
        rows = cur.fetchall()

    for row in rows:
        if row["attachment"]:
            row["attachment"] = json.loads(row["attachment"])
        row["created_at"] = row["created_at"].isoformat()
    return rows
