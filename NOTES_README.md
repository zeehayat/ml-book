# Regression Guide Notes

The guide supports persistent offline notes through a local SQLite database.

## Start

Double-click:

```text
start_notes_server.bat
```

Or run:

```powershell
python notes_server.py
```

Then open:

```text
http://127.0.0.1:8765/
```

## Storage

Notes are stored in:

```text
regression_notes.sqlite3
```

The database is created automatically in this folder the first time the server starts.

## Topic Notes

Each guide topic has an `Add note` button beside its heading. Clicking it opens the floating note panel and attaches the note to that topic. The notes list shows the topic name, and saved notes can jump back to their topic.

The floating `New note` button creates a general note when you are not attaching it to a specific heading.

## Backup

Back up this file:

```text
regression_notes.sqlite3
```

The guide also has an `Export JSON` button for a portable text backup.

## Why a Local Server Is Needed

A browser opened directly from an HTML file cannot write a SQLite database into a normal folder. The local Python server receives notes from the page and writes them to SQLite using Python's built-in `sqlite3` module. No third-party Python packages or remote JavaScript libraries are required.
