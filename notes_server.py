from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
GUIDE_FILE = ROOT / "regression_guide.html"
DB_FILE = ROOT / "regression_notes.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_db() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section TEXT NOT NULL DEFAULT 'General',
                topic_anchor TEXT NOT NULL DEFAULT '',
                topic_title TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()}
        if "topic_anchor" not in columns:
            conn.execute("ALTER TABLE notes ADD COLUMN topic_anchor TEXT NOT NULL DEFAULT ''")
        if "topic_title" not in columns:
            conn.execute("ALTER TABLE notes ADD COLUMN topic_title TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_section ON notes(section)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_topic ON notes(topic_anchor)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_updated_at ON notes(updated_at)")


def row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "section": row["section"],
        "topic_anchor": row["topic_anchor"],
        "topic_title": row["topic_title"],
        "title": row["title"],
        "body": row["body"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class NotesHandler(SimpleHTTPRequestHandler):
    server_version = "RegressionNotes/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, status: int, message: str) -> None:
        data = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.path = "/regression_guide.html"
            return super().do_GET()
        if path == "/api/notes":
            return self.list_notes()
        if path == "/api/notes/export":
            return self.export_notes()
        return super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/notes":
            return self.save_note()
        return self.send_text(404, "Not found")

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        prefix = "/api/notes/"
        if path.startswith(prefix):
            raw_id = path[len(prefix) :]
            try:
                note_id = int(raw_id)
            except ValueError:
                return self.send_text(400, "Invalid note id")
            return self.delete_note(note_id)
        return self.send_text(404, "Not found")

    def list_notes(self) -> None:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, section, title, body, created_at, updated_at
                , topic_anchor, topic_title
                FROM notes
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        self.send_json(
            200,
            {
                "database": str(DB_FILE),
                "notes": [row_to_dict(row) for row in rows],
            },
        )

    def save_note(self) -> None:
        try:
            payload = self.read_json_body()
        except json.JSONDecodeError:
            return self.send_text(400, "Invalid JSON")

        note_id = payload.get("id")
        section = str(payload.get("section") or "General").strip()[:120] or "General"
        topic_anchor = str(payload.get("topic_anchor") or "").strip()[:160]
        topic_title = str(payload.get("topic_title") or "").strip()[:240]
        title = str(payload.get("title") or "").strip()[:240]
        body = str(payload.get("body") or "").strip()
        if not title and not body:
            return self.send_text(400, "Title or body is required")

        now = utc_now()
        with sqlite3.connect(DB_FILE) as conn:
            if note_id:
                conn.execute(
                    """
                    UPDATE notes
                    SET section = ?, topic_anchor = ?, topic_title = ?, title = ?, body = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (section, topic_anchor, topic_title, title, body, now, int(note_id)),
                )
                saved_id = int(note_id)
            else:
                cur = conn.execute(
                    """
                    INSERT INTO notes(section, topic_anchor, topic_title, title, body, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (section, topic_anchor, topic_title, title, body, now, now),
                )
                saved_id = int(cur.lastrowid)
        self.send_json(200, {"ok": True, "id": saved_id})

    def delete_note(self, note_id: int) -> None:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.send_json(200, {"ok": True})

    def export_notes(self) -> None:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, section, title, body, created_at, updated_at
                , topic_anchor, topic_title
                FROM notes
                ORDER BY section ASC, topic_title ASC, updated_at DESC, id DESC
                """
            ).fetchall()
        data = json.dumps([row_to_dict(row) for row in rows], indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="regression_notes_export.json"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    if not GUIDE_FILE.exists():
        raise SystemExit(f"Guide file not found: {GUIDE_FILE}")
    init_db()
    host = "127.0.0.1"
    port = 8765
    server = ThreadingHTTPServer((host, port), NotesHandler)
    print(f"Regression guide with SQLite notes: http://{host}:{port}/")
    print(f"SQLite database: {DB_FILE}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
