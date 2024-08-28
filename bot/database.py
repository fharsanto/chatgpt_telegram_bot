import sqlite3
from typing import Optional, Any
import uuid
from datetime import datetime

import config


class Database:
    def __init__(self):
        self.conn = sqlite3.connect(config.sqlite_db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            chat_id INTEGER,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            last_interaction TIMESTAMP,
            first_seen TIMESTAMP,
            current_dialog_id TEXT,
            current_chat_mode TEXT,
            current_model TEXT,
            n_used_tokens TEXT,
            n_generated_images INTEGER,
            n_transcribed_seconds REAL
        )
        """
        )

        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS dialogs (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            chat_mode TEXT,
            start_time TIMESTAMP,
            model TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
        )

        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dialog_id TEXT,
            role TEXT,
            content TEXT,
            FOREIGN KEY (dialog_id) REFERENCES dialogs (id)
        )
        """
        )

        self.conn.commit()

    def check_if_user_exists(self, user_id: int, raise_exception: bool = False):
        self.cursor.execute("SELECT COUNT(*) FROM users WHERE id = ?", (user_id,))
        exists = self.cursor.fetchone()[0] > 0
        if not exists and raise_exception:
            raise ValueError(f"User {user_id} does not exist")
        return exists

    def add_new_user(
        self,
        user_id: int,
        chat_id: int,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
    ):
        if not self.check_if_user_exists(user_id):
            self.cursor.execute(
                """
            INSERT INTO users (id, chat_id, username, first_name, last_name, last_interaction, first_seen, current_chat_mode, current_model, n_used_tokens, n_generated_images, n_transcribed_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    chat_id,
                    username,
                    first_name,
                    last_name,
                    datetime.now(),
                    datetime.now(),
                    "assistant",
                    config.models["available_text_models"][0],
                    "{}",
                    0,
                    0.0,
                ),
            )
            self.conn.commit()

    def start_new_dialog(self, user_id: int):
        self.check_if_user_exists(user_id, raise_exception=True)

        dialog_id = str(uuid.uuid4())
        chat_mode = self.get_user_attribute(user_id, "current_chat_mode")
        model = self.get_user_attribute(user_id, "current_model")

        self.cursor.execute(
            """
        INSERT INTO dialogs (id, user_id, chat_mode, start_time, model)
        VALUES (?, ?, ?, ?, ?)
        """,
            (dialog_id, user_id, chat_mode, datetime.now(), model),
        )

        self.cursor.execute(
            "UPDATE users SET current_dialog_id = ? WHERE id = ?", (dialog_id, user_id)
        )
        self.conn.commit()

        return dialog_id

    def get_user_attribute(self, user_id: int, key: str):
        self.check_if_user_exists(user_id, raise_exception=True)
        self.cursor.execute(f"SELECT {key} FROM users WHERE id = ?", (user_id,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def set_user_attribute(self, user_id: int, key: str, value: Any):
        self.check_if_user_exists(user_id, raise_exception=True)
        self.cursor.execute(
            f"UPDATE users SET {key} = ? WHERE id = ?", (value, user_id)
        )
        self.conn.commit()

    def update_n_used_tokens(
        self, user_id: int, model: str, n_input_tokens: int, n_output_tokens: int
    ):
        n_used_tokens = self.get_user_attribute(user_id, "n_used_tokens")
        n_used_tokens = eval(n_used_tokens) if n_used_tokens else {}

        if model in n_used_tokens:
            n_used_tokens[model]["n_input_tokens"] += n_input_tokens
            n_used_tokens[model]["n_output_tokens"] += n_output_tokens
        else:
            n_used_tokens[model] = {
                "n_input_tokens": n_input_tokens,
                "n_output_tokens": n_output_tokens,
            }

        self.set_user_attribute(user_id, "n_used_tokens", str(n_used_tokens))

    def get_dialog_messages(self, user_id: int, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id")

        self.cursor.execute(
            "SELECT role, content FROM messages WHERE dialog_id = ? ORDER BY id",
            (dialog_id,),
        )
        messages = self.cursor.fetchall()
        return [{"role": role, "content": content} for role, content in messages]

    def set_dialog_messages(
        self, user_id: int, dialog_messages: list, dialog_id: Optional[str] = None
    ):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id")

        self.cursor.execute("DELETE FROM messages WHERE dialog_id = ?", (dialog_id,))
        for message in dialog_messages:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = (
                    message.get("content")
                    or message.get("user")
                    or message.get("bot", "")
                )
            elif isinstance(message, str):
                role = "user"
                content = message
            else:
                continue  # Skip invalid message formats

            self.cursor.execute(
                "INSERT INTO messages (dialog_id, role, content) VALUES (?, ?, ?)",
                (dialog_id, role, content),
            )
        self.conn.commit()

    def close(self):
        self.conn.close()
