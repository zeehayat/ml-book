from __future__ import annotations

import json
import sqlite3
import re
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import markdown

ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "regression_notes.sqlite3"

GUIDE_DESCRIPTIONS = {
    "regression": "Complete 10-chapter Zero-to-Research ML textbook integrating basics, OLS, GLMs, Tensors, Autograd, Convex Optimization, SVMs, Trees, PCA, Spectral Clustering, and Neural Networks.",
    "generalizedlinearmodelsmathematicalandpython": "The exponential family, link functions, IRLS, deviance, Poisson, binomial, and Gamma regression.",
    "nexttopicagentbrief": "Self-renewing brief outlining the Zero-to-Research ML curriculum guidelines and upcoming topics.",
    "notesreadme": "Instructions and guidelines for using the local offline SQLite notes server."
}

HUB_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zero-to-Research ML Curriculum Hub</title>
  <style>
    :root {
      --bg: #0b1329;
      --surface: #1c2541;
      --surface-hover: #222b4e;
      --border: #3a506b;
      --accent: #5bc0be;
      --accent-glow: rgba(91, 192, 190, 0.15);
      --text: #ffffff;
      --text-muted: #a5b4fc;
      --font: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 60px 20px 40px;
      text-align: center;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(28,37,65,0.8) 0%, rgba(11,19,41,0.8) 100%);
    }
    header h1 {
      font-size: clamp(28px, 4vw, 42px);
      margin: 0 0 12px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #a5b4fc 0%, var(--accent) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    header p {
      color: var(--text-muted);
      max-width: 600px;
      margin: 0 auto;
      font-size: 16px;
    }
    main {
      flex: 1;
      max-width: 1000px;
      width: 100%;
      margin: 0 auto;
      padding: 40px 20px 80px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 24px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 4px 20px rgba(0,0,0,0.2);
      text-decoration: none;
      color: inherit;
    }
    .card:hover {
      transform: translateY(-5px);
      border-color: var(--accent);
      box-shadow: 0 8px 30px var(--accent-glow);
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 20px;
      font-weight: 700;
      color: #fff;
    }
    .card p {
      margin: 0 0 20px;
      color: var(--text-muted);
      font-size: 14px;
      flex: 1;
    }
    .card-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      border-top: 1px solid rgba(255,255,255,0.08);
      padding-top: 16px;
    }
    .notes-badge {
      background: rgba(91, 192, 190, 0.1);
      color: var(--accent);
      padding: 4px 10px;
      border-radius: 99px;
      font-weight: 600;
    }
    .open-btn {
      color: var(--accent);
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .open-btn::after {
      content: "→";
      transition: transform 0.2s;
    }
    .card:hover .open-btn::after {
      transform: translateX(4px);
    }
    footer {
      text-align: center;
      padding: 24px;
      color: var(--text-muted);
      font-size: 12px;
      border-top: 1px solid var(--border);
      background: rgba(28,37,65,0.2);
    }
  </style>
</head>
<body>
  <header>
    <h1>Zero-to-Research ML Curriculum Hub</h1>
    <p>A step-by-step machine learning sequence. Select a guide to study with local SQLite-powered notes.</p>
  </header>
  <main>
    <div class="grid">
      <!-- Cards will be populated dynamically -->
    </div>
  </main>
  <footer>
    Curriculum Hub • Local SQLite Database
  </footer>
</body>
</html>
"""

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def normalize_guide_name(name: str) -> str:
    if name.endswith(".html") or name.endswith(".md"):
        name = name[:-5]
    name = name.lower().replace("_", "").replace("-", "")
    if name.endswith("guide"):
        name = name[:-5]
    return name or "general"

def get_guide_title(filename: str) -> str:
    stem = Path(filename).stem
    if stem == "Regression_Mathematical_and_Python_Guide":
        return "Regression: A Mathematical and Python Guide"
    if stem == "Generalized_Linear_Models_Mathematical_and_Python_Guide":
        return "Generalized Linear Models: A Mathematical and Python Guide"
    if stem == "NEXT_TOPIC_AGENT_BRIEF":
        return "Next Topic Agent Brief"
    return stem.replace("_", " ")

def init_db() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guide TEXT NOT NULL DEFAULT 'regression',
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
        if "guide" not in columns:
            conn.execute("ALTER TABLE notes ADD COLUMN guide TEXT NOT NULL DEFAULT 'regression'")
            
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_section ON notes(section)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_topic ON notes(topic_anchor)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_guide ON notes(guide)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_updated_at ON notes(updated_at)")

def row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "guide": row["guide"] if "guide" in row.keys() else "regression",
        "section": row["section"],
        "topic_anchor": row["topic_anchor"],
        "topic_title": row["topic_title"],
        "title": row["title"],
        "body": row["body"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

def find_markdown_file(stem: str) -> Path | None:
    norm = normalize_guide_name(stem)
    for f in ROOT.glob("*.md"):
        f_norm = normalize_guide_name(f.name)
        if f_norm == norm:
            return f
    for f in ROOT.glob("*.md"):
        f_norm = normalize_guide_name(f.name)
        if f_norm.startswith(norm) or norm.startswith(f_norm) or f_norm in norm or norm in f_norm:
            return f
    return None

def get_note_counts() -> dict[str, int]:
    counts = {"regression": 0}
    for f in ROOT.glob("*.md"):
        counts[normalize_guide_name(f.name)] = 0
    try:
        with sqlite3.connect(DB_FILE) as conn:
            rows = conn.execute("SELECT guide, COUNT(*) FROM notes GROUP BY guide").fetchall()
            for row in rows:
                counts[normalize_guide_name(row[0])] = row[1]
    except Exception:
        pass
    return counts

def render_markdown_guide(md_file: Path) -> str:
    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    md = markdown.Markdown(extensions=["fenced_code", "tables", "toc"])
    html_body = md.convert(text)
    toc_html = md.toc

    title = md_file.stem.replace("_", " ")
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    guide_key = normalize_guide_name(md_file.name)

    html_template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{TITLE}}</title>
  <style>
    :root {
      --ink: #1c232b;
      --muted: #5b6673;
      --line: #d9dee5;
      --paper: #ffffff;
      --soft: #f5f7fa;
      --accent: #0f766e;
      --accent2: #8b3a3a;
      --code: #111827;
      --codebg: #f0f3f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
      line-height: 1.62;
    }
    header {
      padding: 42px max(24px, calc((100vw - 1120px) / 2)) 28px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f7fbfb 0%, #fff 100%);
    }
    main {
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 24px 80px;
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 34px;
    }
    nav {
      position: sticky;
      top: 0;
      align-self: start;
      max-height: 100vh;
      overflow: auto;
      padding: 18px 0;
      border-right: 1px solid var(--line);
    }
    nav a, nav .toc a {
      display: block;
      color: var(--ink);
      text-decoration: none;
      padding: 5px 16px 5px 0;
      font-size: 14px;
    }
    nav a:hover, nav .toc a:hover { color: var(--accent); }
    nav .toc ul {
      list-style: none;
      padding-left: 0;
      margin: 0;
    }
    nav .toc li {
      margin: 0;
      padding: 0;
    }
    nav .toc ul ul {
      padding-left: 12px;
    }
    h1 {
      font-size: clamp(32px, 4vw, 54px);
      line-height: 1.05;
      margin: 0 0 12px;
      letter-spacing: 0;
    }
    h2 {
      margin: 44px 0 12px;
      padding-top: 10px;
      border-top: 2px solid var(--line);
      font-size: 30px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    h3 {
      margin: 28px 0 8px;
      font-size: 21px;
      letter-spacing: 0;
    }
    h4 { margin: 20px 0 6px; font-size: 17px; }
    p, li { max-width: 78ch; }
    .lede { max-width: 86ch; color: var(--muted); font-size: 18px; }
    .box {
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px 18px;
      margin: 16px 0;
    }
    .warn { border-left: 5px solid var(--accent2); }
    .key { border-left: 5px solid var(--accent); }
    .review-note {
      border-left: 5px solid #5b5f97;
      background: #f7f7fc;
    }
    code, pre { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }
    code { background: var(--codebg); padding: 1px 4px; border-radius: 4px; }
    pre {
      overflow: auto;
      background: var(--code);
      color: #f8fafc;
      padding: 16px;
      border-radius: 8px;
      line-height: 1.45;
      font-size: 13px;
    }
    pre code { background: transparent; padding: 0; color: inherit; }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 16px 0 22px;
      font-size: 14px;
    }
    th, td { border: 1px solid var(--line); padding: 9px 10px; vertical-align: top; }
    th { background: var(--soft); text-align: left; }
    .formula {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      background: #fbfbfd;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      overflow: auto;
      white-space: pre-wrap;
    }
    .exercise {
      border-left: 4px solid var(--accent);
      padding: 9px 14px;
      background: #f6fbfa;
      margin: 12px 0 20px;
    }
    .term {
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 18px;
    }
    .expand-toolbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 14px 0 18px;
    }
    .expand-toolbar button {
      border: 1px solid #94a3b8;
      border-radius: 6px;
      padding: 8px 11px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }
    details.deep-dive {
      max-width: 850px;
      margin: 12px 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    details.deep-dive > summary {
      cursor: pointer;
      padding: 12px 14px;
      background: #f8fafc;
      font-weight: 700;
      list-style-position: inside;
    }
    details.deep-dive[open] > summary {
      border-bottom: 1px solid var(--line);
      background: #eef7f6;
      color: #0f5f59;
    }
    .deep-dive-body {
      padding: 14px 16px 16px;
    }
    .deep-dive-body h4:first-child { margin-top: 0; }
    .mental-model {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 14px 0;
      max-width: 850px;
    }
    .mental-model > div {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }
    .mental-model h4 { margin-top: 0; }
    .notes-panel {
      background: #fbfcfd;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 20px 0 28px;
      max-width: 850px;
    }
    .notes-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
    }
    .notes-panel label {
      display: block;
      font-weight: 650;
      margin-bottom: 5px;
    }
    .notes-panel input,
    .notes-panel select,
    .notes-panel textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    .notes-panel textarea {
      min-height: 150px;
      resize: vertical;
    }
    .notes-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .notes-actions button {
      border: 1px solid #94a3b8;
      border-radius: 6px;
      padding: 8px 11px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }
    .notes-actions button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    .notes-status {
      color: var(--muted);
      font-size: 14px;
      margin-top: 10px;
    }
    .notes-list {
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }
    .topic-note-button {
      margin-left: 8px;
      border: 1px solid var(--accent);
      border-radius: 999px;
      background: #fff;
      color: var(--accent);
      font: 12px system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      padding: 3px 8px;
      vertical-align: middle;
      cursor: pointer;
    }
    .floating-note-button {
      position: fixed;
      right: 22px;
      bottom: 22px;
      z-index: 20;
      border: 1px solid var(--accent);
      border-radius: 999px;
      background: var(--accent);
      color: #fff;
      font: 700 15px system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      padding: 12px 16px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.22);
      cursor: pointer;
    }
    .floating-note-panel {
      position: fixed;
      right: 22px;
      bottom: 78px;
      z-index: 21;
      width: min(430px, calc(100vw - 44px));
      max-height: calc(100vh - 110px);
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 18px 50px rgba(15, 23, 42, 0.24);
      padding: 16px;
    }
    .floating-note-panel[hidden] {
      display: none;
    }
    .floating-note-panel header {
      padding: 0 0 10px;
      border: 0;
      background: transparent;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .floating-note-panel h3 {
      margin: 0;
      font-size: 18px;
    }
    .floating-note-panel label {
      display: block;
      margin: 10px 0 5px;
      font-weight: 650;
    }
    .floating-note-panel input,
    .floating-note-panel select,
    .floating-note-panel textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
    }
    .floating-note-panel textarea {
      min-height: 170px;
      resize: vertical;
    }
    .floating-topic-context {
      color: var(--muted);
      font-size: 13px;
      margin: 0 0 6px;
    }
    .close-note-panel {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      font: inherit;
      padding: 4px 8px;
    }
    .note-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }
    .note-card header {
      padding: 0;
      border: 0;
      background: transparent;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }
    .note-card h4 {
      margin: 0;
      font-size: 16px;
    }
    .note-card .meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .note-card p {
      margin: 8px 0 0;
      white-space: pre-wrap;
    }
    .home-link {
      display: inline-block;
      margin-bottom: 20px;
      color: var(--accent);
      text-decoration: none;
      font-weight: bold;
    }
    .home-link:hover {
      text-decoration: underline;
    }
    @media (max-width: 720px) {
      .notes-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 880px) {
      main { display: block; }
      nav { position: static; max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
<header>
  <a href="/" class="home-link">← Hub Dashboard</a>
  <h1>{{TITLE}}</h1>
</header>
<main>
  <nav aria-label="Table of contents">
    <a href="#notes">Notes</a>
    {{TOC}}
  </nav>
  <article>
    <section id="notes">
      <h2>Offline Notes</h2>
      <p>Use this panel when the guide is opened through the local notes server. Notes are stored in <code>regression_notes.sqlite3</code> in the same folder as this HTML file.</p>
      <div class="notes-panel" data-notes-app>
        <div class="notes-grid">
          <div>
            <label for="note-section">Section</label>
            <select id="note-section">
              <option>General</option>
            </select>
          </div>
          <div>
            <label for="note-title">Title</label>
            <input id="note-title" type="text" placeholder="Example: Key concept definition">
          </div>
        </div>
        <label for="note-body" style="margin-top: 12px;">Note</label>
        <textarea id="note-body" placeholder="Write your note, question, derivation, or teaching prompt here."></textarea>
        <div class="notes-actions">
          <button class="primary" type="button" id="save-note">Save note</button>
          <button type="button" id="refresh-notes">Refresh</button>
          <button type="button" id="export-notes">Export JSON</button>
          <button type="button" id="clear-note-form">Clear form</button>
        </div>
        <div class="notes-status" id="notes-status">Notes save to SQLite when the local server is running.</div>
        <div class="notes-list" id="notes-list" aria-live="polite"></div>
      </div>
    </section>

    {{BODY}}
  </article>
</main>

<button class="floating-note-button" type="button" id="open-note-panel">New note</button>
<aside class="floating-note-panel" id="floating-note-panel" hidden aria-label="Create topic note">
  <header>
    <h3>Topic Note</h3>
    <button class="close-note-panel" type="button" id="close-note-panel">Close</button>
  </header>
  <p class="floating-topic-context" id="floating-topic-context">Topic: General</p>
  <label for="floating-note-section">Section</label>
  <select id="floating-note-section">
    <option>General</option>
  </select>
  <label for="floating-note-title">Title</label>
  <input id="floating-note-title" type="text" placeholder="My understanding of this topic">
  <label for="floating-note-body">Note</label>
  <textarea id="floating-note-body" placeholder="Explain the topic in your own words, add questions, derivations, or examples."></textarea>
  <div class="notes-actions">
    <button class="primary" type="button" id="floating-save-note">Save note</button>
    <button type="button" id="floating-clear-note">Clear</button>
  </div>
  <div class="notes-status" id="floating-notes-status">Notes save to SQLite when the local server is running.</div>
</aside>

<script>
(() => {
  const guideKey = "{{GUIDE_KEY}}";
  
  const sectionEl = document.getElementById("note-section");
  const titleEl = document.getElementById("note-title");
  const bodyEl = document.getElementById("note-body");
  const saveBtn = document.getElementById("save-note");
  const refreshBtn = document.getElementById("refresh-notes");
  const clearBtn = document.getElementById("clear-note-form");
  const exportBtn = document.getElementById("export-notes");
  const listEl = document.getElementById("notes-list");
  const statusEl = document.getElementById("notes-status");

  const openPanelBtn = document.getElementById("open-note-panel");
  const closePanelBtn = document.getElementById("close-note-panel");
  const floatingPanel = document.getElementById("floating-note-panel");
  const floatingContextEl = document.getElementById("floating-topic-context");
  const floatingSectionEl = document.getElementById("floating-note-section");
  const floatingTitleEl = document.getElementById("floating-note-title");
  const floatingBodyEl = document.getElementById("floating-note-body");
  const floatingSaveBtn = document.getElementById("floating-save-note");
  const floatingClearBtn = document.getElementById("floating-clear-note");
  const floatingStatusEl = document.getElementById("floating-notes-status");

  let editingId = null;
  let floatingEditingId = null;
  let currentTopic = { section: "General", topic_anchor: "", topic_title: "General" };
  const topicButtons = new Map();

  function setStatus(message) {
    statusEl.textContent = message;
    if (floatingStatusEl) floatingStatusEl.textContent = message;
  }

  function escapeText(value) {
    return String(value ?? "");
  }

  function slugify(value) {
    return String(value)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 90) || "topic";
  }

  function cleanHeadingText(heading) {
    return Array.from(heading.childNodes)
      .filter((node) => node.nodeType === Node.TEXT_NODE)
      .map((node) => node.textContent)
      .join(" ")
      .replace(/\\s+/g, " ")
      .trim();
  }

  function sectionNameFor(heading) {
    if (heading.tagName === "H2") {
      return cleanHeadingText(heading).replace(/^\\d+\\.\\s*/, "") || "General";
    }
    let prev = heading.previousElementSibling;
    while (prev) {
      if (prev.tagName === "H2") {
        return cleanHeadingText(prev).replace(/^\\d+\\.\\s*/, "") || "General";
      }
      prev = prev.previousElementSibling;
    }
    return "General";
  }

  function populateSectionOptions(sections) {
    const selectElements = [sectionEl, floatingSectionEl];
    selectElements.forEach(select => {
      if (!select) return;
      select.innerHTML = '<option>General</option>';
      sections.forEach(sec => {
        if (sec && sec !== "General") {
          const opt = document.createElement("option");
          opt.textContent = sec;
          select.appendChild(opt);
        }
      });
    });
  }

  function ensureTopicControls() {
    const headings = document.querySelectorAll("article h2, article h3");
    const used = new Set();
    const sections = new Set();

    headings.forEach((heading) => {
      const topicTitle = cleanHeadingText(heading);
      if (!topicTitle) return;
      if (heading.closest("#notes")) return;
      
      const secName = sectionNameFor(heading);
      sections.add(secName);

      if (!heading.id) {
        let base = slugify(topicTitle);
        let id = base;
        let i = 2;
        while (document.getElementById(id) || used.has(id)) {
          id = `${base}-${i}`;
          i += 1;
        }
        heading.id = id;
        used.add(id);
      }

      const button = document.createElement("button");
      button.type = "button";
      button.className = "topic-note-button";
      button.textContent = "Add note";
      button.title = `Add a note for: ${topicTitle}`;
      button.addEventListener("click", () => {
        openFloatingPanel({
          section: secName,
          topic_anchor: heading.id,
          topic_title: topicTitle
        });
      });
      heading.appendChild(button);
      topicButtons.set(heading.id, button);
    });
    
    populateSectionOptions(Array.from(sections));
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }
    return response.json();
  }

  function clearForm() {
    editingId = null;
    titleEl.value = "";
    bodyEl.value = "";
    sectionEl.value = "General";
    saveBtn.textContent = "Save note";
  }

  function clearFloatingForm(keepTopic = true) {
    floatingEditingId = null;
    floatingTitleEl.value = "";
    floatingBodyEl.value = "";
    floatingSaveBtn.textContent = "Save note";
    if (!keepTopic) {
      currentTopic = { section: "General", topic_anchor: "", topic_title: "General" };
      floatingSectionEl.value = "General";
      floatingContextEl.textContent = "Topic: General";
    }
  }

  function openFloatingPanel(topic, note = null) {
    currentTopic = {
      section: topic.section || "General",
      topic_anchor: topic.topic_anchor || "",
      topic_title: topic.topic_title || "General"
    };
    floatingPanel.hidden = false;
    floatingSectionEl.value = currentTopic.section;
    floatingContextEl.textContent = `Topic: ${currentTopic.topic_title}`;

    if (note) {
      floatingEditingId = note.id;
      floatingTitleEl.value = note.title || "";
      floatingBodyEl.value = note.body || "";
      floatingSaveBtn.textContent = "Update note";
    } else {
      clearFloatingForm(true);
    }
    floatingTitleEl.focus();
  }

  function updateTopicButtonCounts(notes) {
    const counts = new Map();
    for (const note of notes) {
      if (!note.topic_anchor) continue;
      counts.set(note.topic_anchor, (counts.get(note.topic_anchor) || 0) + 1);
    }
    for (const [anchor, button] of topicButtons.entries()) {
      const count = counts.get(anchor) || 0;
      button.textContent = count ? `Add note (${count})` : "Add note";
    }
  }

  function renderNotes(notes) {
    listEl.innerHTML = "";
    updateTopicButtonCounts(notes);
    if (!notes.length) {
      const empty = document.createElement("p");
      empty.className = "notes-status";
      empty.textContent = "No notes saved yet.";
      listEl.appendChild(empty);
      return;
    }

    for (const note of notes) {
      const card = document.createElement("article");
      card.className = "note-card";

      const header = document.createElement("header");
      const title = document.createElement("h4");
      title.textContent = note.title || "(Untitled note)";
      const meta = document.createElement("span");
      meta.className = "meta";
      const topic = note.topic_title ? ` > ${note.topic_title}` : "";
      meta.textContent = `${note.section}${topic} | ${note.updated_at}`;
      header.append(title, meta);

      const body = document.createElement("p");
      body.textContent = escapeText(note.body);

      const actions = document.createElement("div");
      actions.className = "notes-actions";

      const edit = document.createElement("button");
      edit.type = "button";
      edit.textContent = "Edit";
      edit.addEventListener("click", () => {
        openFloatingPanel({
          section: note.section || "General",
          topic_anchor: note.topic_anchor || "",
          topic_title: note.topic_title || "General"
        }, note);
      });

      if (note.topic_anchor) {
        const jump = document.createElement("button");
        jump.type = "button";
        jump.textContent = "Go to topic";
        jump.addEventListener("click", () => {
          document.getElementById(note.topic_anchor)?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
        actions.appendChild(jump);
      }

      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Delete";
      remove.addEventListener("click", async () => {
        if (!confirm("Delete this note from the SQLite database?")) return;
        try {
          await api(`/api/notes/${note.id}`, { method: "DELETE" });
          setStatus("Note deleted.");
          await loadNotes();
        } catch (err) {
          setStatus(`Delete failed: ${err.message}`);
        }
      });

      actions.append(edit, remove);
      card.append(header, body, actions);
      listEl.appendChild(card);
    }
  }

  async function loadNotes() {
    try {
      const data = await api(`/api/notes?guide=${guideKey}`);
      renderNotes(data.notes || []);
      setStatus(`SQLite notes ready. Database: ${data.database}`);
    } catch (err) {
      renderNotes([]);
      setStatus("Notes are offline.");
    }
  }

  saveBtn.addEventListener("click", async () => {
    const title = titleEl.value.trim();
    const body = bodyEl.value.trim();
    if (!title && !body) {
      setStatus("Write a title or note body before saving.");
      return;
    }

    try {
      await api("/api/notes", {
        method: "POST",
        body: JSON.stringify({
          id: editingId,
          guide: guideKey,
          section: sectionEl.value,
          topic_anchor: "",
          topic_title: "",
          title,
          body
        })
      });
      clearForm();
      setStatus("Note saved to SQLite.");
      await loadNotes();
    } catch (err) {
      setStatus(`Save failed: ${err.message}`);
    }
  });

  refreshBtn.addEventListener("click", loadNotes);
  clearBtn.addEventListener("click", clearForm);
  exportBtn.addEventListener("click", () => {
    window.location.href = `/api/notes/export?guide=${guideKey}`;
  });
  openPanelBtn.addEventListener("click", () => openFloatingPanel({
    section: "General",
    topic_anchor: "",
    topic_title: "General"
  }));
  closePanelBtn.addEventListener("click", () => {
    floatingPanel.hidden = true;
  });
  floatingClearBtn.addEventListener("click", () => clearFloatingForm(true));
  floatingSaveBtn.addEventListener("click", async () => {
    const title = floatingTitleEl.value.trim();
    const body = floatingBodyEl.value.trim();
    if (!title && !body) {
      setStatus("Write a title or note body before saving.");
      return;
    }

    try {
      await api("/api/notes", {
        method: "POST",
        body: JSON.stringify({
          id: floatingEditingId,
          guide: guideKey,
          section: floatingSectionEl.value || currentTopic.section,
          topic_anchor: currentTopic.topic_anchor,
          topic_title: currentTopic.topic_title,
          title,
          body
        })
      });
      clearFloatingForm(true);
      floatingPanel.hidden = true;
      setStatus("Topic note saved to SQLite.");
      await loadNotes();
    } catch (err) {
      setStatus(`Save failed: ${err.message}`);
    }
  });

  ensureTopicControls();
  loadNotes();
})();
</script>
</body>
</html>
"""

    return (
        html_template.replace("{{TITLE}}", title)
        .replace("{{TOC}}", toc_html)
        .replace("{{BODY}}", html_body)
        .replace("{{GUIDE_KEY}}", guide_key)
    )


class NotesHandler(SimpleHTTPRequestHandler):
    server_version = "MLCurriculumNotes/1.0"

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

    def serve_hub(self) -> None:
        counts = get_note_counts()
        cards_html = []
        
        # OLS static Regression Guide
        reg_count = counts.get("regression", 0)
        reg_desc = GUIDE_DESCRIPTIONS.get("regression", "")
        cards_html.append(f"""
        <a class="card" href="/regression_guide.html">
          <div>
            <h2>Zero-to-Research Machine Learning Textbook (Complete)</h2>
            <p>{reg_desc}</p>
          </div>
          <div class="card-meta">
            <span class="notes-badge">{reg_count} notes</span>
            <span class="open-btn">Open Guide</span>
          </div>
        </a>
        """)

        # Other markdown files
        for f in sorted(ROOT.glob("*.md")):
            norm_name = normalize_guide_name(f.name)
            title = get_guide_title(f.name)
            desc = GUIDE_DESCRIPTIONS.get(norm_name, "Learn about this topic in the zero-to-research ML curriculum.")
            count = counts.get(norm_name, 0)
            href = f"{f.stem}.html"
            
            cards_html.append(f"""
            <a class="card" href="/{href}">
              <div>
                <h2>{title}</h2>
                <p>{desc}</p>
              </div>
              <div class="card-meta">
                <span class="notes-badge">{count} notes</span>
                <span class="open-btn">Open Guide</span>
              </div>
            </a>
            """)

        template = HUB_TEMPLATE.replace("<!-- Cards will be populated dynamically -->", "\n".join(cards_html))
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(template.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(template.encode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            return self.serve_hub()
        if path == "/api/notes":
            return self.list_notes()
        if path == "/api/notes/export":
            return self.export_notes()
            
        if path.endswith(".html"):
            stem = Path(path).stem
            static_file = ROOT / f"{stem}.html"
            if static_file.exists():
                return super().do_GET()
                
            md_file = find_markdown_file(stem)
            if md_file and md_file.exists():
                html_content = render_markdown_guide(md_file)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_content.encode("utf-8"))))
                self.end_headers()
                self.wfile.write(html_content.encode("utf-8"))
                return
                
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
        query = urlparse(self.path).query
        params = parse_qs(query)
        guide = params.get("guide", [None])[0]
        if not guide:
            referer = self.headers.get("Referer", "")
            if referer:
                ref_path = urlparse(referer).path
                guide = Path(ref_path).stem or "regression"
            else:
                guide = "regression"
        
        guide = normalize_guide_name(guide)
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, guide, section, title, body, created_at, updated_at
                , topic_anchor, topic_title
                FROM notes
                WHERE guide = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (guide,),
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

        guide = payload.get("guide")
        if not guide:
            referer = self.headers.get("Referer", "")
            if referer:
                ref_path = urlparse(referer).path
                guide = Path(ref_path).stem or "regression"
            else:
                guide = "regression"
                
        guide = normalize_guide_name(guide)
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
                    INSERT INTO notes(guide, section, topic_anchor, topic_title, title, body, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (guide, section, topic_anchor, topic_title, title, body, now, now),
                )
                saved_id = int(cur.lastrowid)
        self.send_json(200, {"ok": True, "id": saved_id})

    def delete_note(self, note_id: int) -> None:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.send_json(200, {"ok": True})

    def export_notes(self) -> None:
        query = urlparse(self.path).query
        params = parse_qs(query)
        guide = params.get("guide", [None])[0]
        if not guide:
            referer = self.headers.get("Referer", "")
            if referer:
                ref_path = urlparse(referer).path
                guide = Path(ref_path).stem or "regression"
            else:
                guide = "regression"
                
        guide = normalize_guide_name(guide)
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, guide, section, title, body, created_at, updated_at
                , topic_anchor, topic_title
                FROM notes
                WHERE guide = ?
                ORDER BY section ASC, topic_title ASC, updated_at DESC, id DESC
                """,
                (guide,),
            ).fetchall()
        data = json.dumps([row_to_dict(row) for row in rows], indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{guide}_notes_export.json"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    import sys
    init_db()
    host = "127.0.0.1"
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    server = ThreadingHTTPServer((host, port), NotesHandler)
    print(f"ML Curriculum Notes Hub: http://{host}:{port}/")
    print(f"SQLite database: {DB_FILE}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()

