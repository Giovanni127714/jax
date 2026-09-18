"""MySQL persistence: users and chat message history.

Connects to the MySQL server bundled with Laragon (root / no password by
default). Override via DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME env vars
if your setup differs.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pymysql
from pymysql.cursors import DictCursor

DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "jax_chat")


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


def init_db() -> None:
    """Creates the database and tables if they don't exist yet."""
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
                CREATE TABLE IF NOT EXISTS messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    role ENUM('user', 'assistant') NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    attachment JSON NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user_created (user_id, created_at),
                    CONSTRAINT fk_messages_user FOREIGN KEY (user_id)
                        REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------


def create_user(username: str, password_hash: str) -> int:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, password_hash),
            )
            return cur.lastrowid
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            return cur.fetchone()
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cur.fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------


def save_message(
    user_id: int, role: str, content: str, attachment: Optional[Dict[str, Any]] = None
) -> int:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (user_id, role, content, attachment) "
                "VALUES (%s, %s, %s, %s)",
                (
                    user_id,
                    role,
                    content,
                    json.dumps(attachment) if attachment else None,
                ),
            )
            return cur.lastrowid
    finally:
        conn.close()


def get_messages(user_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, attachment, created_at FROM messages "
                "WHERE user_id = %s ORDER BY id ASC LIMIT %s",
                (user_id, limit),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    for row in rows:
        if row["attachment"]:
            row["attachment"] = json.loads(row["attachment"])
        row["created_at"] = row["created_at"].isoformat()
    return rows
