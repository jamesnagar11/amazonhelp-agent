"""
Memory management for multi-chat LangGraph sessions.

Provides:
  - SqliteSaver-based checkpointing (thread_id isolation)
  - Thread CRUD: create / list / delete
  - Short-term memory: last MAX_MESSAGES messages (or MAX_TOKENS tokens)
  - Long-term memory: rolling summaries via DeepSeek
  - Human/AI message dicts for DB persistence
"""

from __future__ import annotations

import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SQLITE_PATH = os.getenv("SQLITE_PATH", "./checkpoints/chat_checkpoints.db")
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "10"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))

# Ensure the directory exists
Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)

# ── Thread metadata table (separate from langgraph checkpoints) ─────────────

METADATA_DB = str(Path(SQLITE_PATH).parent / "thread_metadata.db")


def _get_meta_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(METADATA_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            messages TEXT,
            summary_list TEXT,
            iteration_count INTEGER DEFAULT 0
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            message_index INTEGER,
            rating TEXT,
            feedback_text TEXT,
            created_at TEXT
        )"""
    )
    conn.commit()
    return conn


# ── Public API ───────────────────────────────────────────────────────────────

def list_threads() -> list[dict]:
    """Return all thread metadata sorted by updated_at descending."""
    with _get_meta_conn() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at, iteration_count "
            "FROM threads ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {
            "thread_id": r[0],
            "title": r[1] or "New Chat",
            "created_at": r[2],
            "updated_at": r[3],
            "iteration_count": r[4],
        }
        for r in rows
    ]


def create_thread(thread_id: str, title: str = "New Chat") -> dict:
    """Create a new thread entry."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    with _get_meta_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO threads (thread_id, title, created_at, updated_at, messages, summary_list, iteration_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (thread_id, title, now, now, "[]", "[]", 0),
        )
        conn.commit()
    return get_thread(thread_id)


def get_thread(thread_id: str) -> dict | None:
    """Load full thread data including messages and summaries."""
    with _get_meta_conn() as conn:
        row = conn.execute(
            "SELECT thread_id, title, created_at, updated_at, messages, summary_list, iteration_count "
            "FROM threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "thread_id": row[0],
        "title": row[1],
        "created_at": row[2],
        "updated_at": row[3],
        "messages": json.loads(row[4] or "[]"),
        "summary_list": json.loads(row[5] or "[]"),
        "iteration_count": row[6],
    }


def delete_thread(thread_id: str) -> bool:
    """Delete a thread and its langgraph checkpoints."""
    with _get_meta_conn() as conn:
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        conn.commit()
    # Also remove from langgraph checkpoint DB if it exists
    try:
        lc_conn = sqlite3.connect(SQLITE_PATH)
        lc_conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        lc_conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
        lc_conn.commit()
        lc_conn.close()
    except Exception:
        pass
    return True


def save_thread_state(
    thread_id: str,
    messages: list[dict],
    summary_list: list[str],
    iteration_count: int,
    title: str | None = None,
) -> None:
    """Persist updated messages + summaries after a graph run."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    with _get_meta_conn() as conn:
        # Auto-title from first human message if not set
        if title is None:
            row = conn.execute(
                "SELECT title FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if row and (row[0] == "New Chat" or row[0] is None):
                human_msgs = [m for m in messages if m.get("role") == "human"]
                if human_msgs:
                    raw = human_msgs[0]["content"][:50]
                    title = raw if len(raw) < 50 else raw + "…"
                else:
                    title = "New Chat"
            else:
                title = row[0] if row else "New Chat"

        conn.execute(
            "INSERT INTO threads (thread_id, title, created_at, updated_at, messages, summary_list, iteration_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET "
            "  title=excluded.title, updated_at=excluded.updated_at, "
            "  messages=excluded.messages, summary_list=excluded.summary_list, "
            "  iteration_count=excluded.iteration_count",
            (
                thread_id,
                title,
                now,
                now,
                json.dumps(messages),
                json.dumps(summary_list),
                iteration_count,
            ),
        )
        conn.commit()


# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Approximate token count using whitespace split (no tiktoken required)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def count_messages_tokens(messages: list[dict]) -> int:
    """Count total tokens across a list of message dicts."""
    return sum(count_tokens(m.get("content", "")) for m in messages)


# ── Sliding window + summarisation ───────────────────────────────────────────

from langsmith import traceable


@traceable(name="trim_messages", tags=["memory", "sliding_window"], metadata={"component": "message_trimmer"})
def trim_messages(
    messages: list[dict],
    summary_list: list[str],
    llm: Any | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Apply sliding window:
      - Keep last MAX_MESSAGES messages OR within MAX_TOKENS — whichever binds first.
      - Older messages are summarised by DeepSeek and appended to summary_list.
    Returns (trimmed_messages, updated_summary_list).
    """
    if len(messages) <= MAX_MESSAGES and count_messages_tokens(messages) <= MAX_TOKENS:
        return messages, summary_list

    # Determine split point: keep last MAX_MESSAGES within token budget
    keep = messages[-MAX_MESSAGES:]
    while count_messages_tokens(keep) > MAX_TOKENS and len(keep) > 2:
        keep = keep[1:]

    overflow = messages[: len(messages) - len(keep)]
    if not overflow:
        return keep, summary_list

    # Summarise overflow with DeepSeek
    summary = _summarise_messages(overflow, llm)
    if summary:
        summary_list = list(summary_list) + [summary]
        logger.info("Summarised %d overflow messages into summary #%d", len(overflow), len(summary_list))

    return keep, summary_list


@traceable(name="summarise_messages", tags=["memory", "llm_summary"], metadata={"component": "chat_summariser"})
def _summarise_messages(messages: list[dict], llm: Any | None = None) -> str:
    """Use DeepSeek (judge LLM) to create a concise summary of old messages."""
    if not messages:
        return ""
    if llm is None:
        from src.utils.llm import get_judge_llm
        llm = get_judge_llm()

    formatted = "\n".join(
        f"{'USER' if m['role'] == 'human' else 'AGENT'}: {m['content']}" for m in messages
    )
    prompt = (
        "You are summarising a conversation between a customer and an Amazon customer support agent.\n"
        "Create a concise but complete summary that preserves all key facts:\n"
        "  - What the customer's issue was\n"
        "  - What solutions were tried\n"
        "  - What the current resolution status is\n"
        "Keep the summary under 150 words.\n\n"
        f"CONVERSATION:\n{formatted}\n\nSUMMARY:"
    )
    try:
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()
    except Exception as e:
        logger.warning("Summarisation failed: %s", e)
        return f"[Previous conversation: {len(messages)} messages]"


def save_feedback(
    thread_id: str,
    message_index: int,
    rating: str,
    feedback_text: str = "",
) -> dict:
    """Save user feedback rating (thumbs_up / thumbs_down) for a specific message turn."""
    import uuid
    from datetime import datetime, timezone

    fid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _get_meta_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (id, thread_id, message_index, rating, feedback_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (fid, thread_id, message_index, rating, feedback_text, now),
        )
        conn.commit()
    return {
        "id": fid,
        "thread_id": thread_id,
        "message_index": message_index,
        "rating": rating,
        "feedback_text": feedback_text,
        "created_at": now,
    }


def get_feedback_summary() -> dict:
    """Return summary statistics of user feedback."""
    with _get_meta_conn() as conn:
        pos = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'positive'").fetchone()[0]
        neg = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'negative'").fetchone()[0]
        recent = conn.execute(
            "SELECT thread_id, message_index, rating, feedback_text, created_at "
            "FROM feedback ORDER BY created_at DESC LIMIT 10"
        ).fetchall()

    return {
        "positive": pos,
        "negative": neg,
        "total": pos + neg,
        "recent": [
            {
                "thread_id": r[0],
                "message_index": r[1],
                "rating": r[2],
                "feedback_text": r[3],
                "created_at": r[4],
            }
            for r in recent
        ],
    }


def build_context_string(summary_list: list[str], messages: list[dict]) -> str:
    """Build a context string combining summaries + recent messages for LLM input with token caps."""
    parts = []
    if summary_list:
        parts.append("=== PREVIOUS CONVERSATION SUMMARIES ===")
        summary_tokens = 0
        for i, s in enumerate(summary_list, 1):
            s_tok = count_tokens(s)
            if summary_tokens + s_tok > 1500:
                parts.append("[Earlier summaries truncated due to token limit]")
                break
            parts.append(f"[Summary {i}]: {s}")
            summary_tokens += s_tok
        parts.append("")

    if messages:
        parts.append("=== RECENT CONVERSATION ===")
        msg_tokens = 0
        recent_msgs = []
        for m in reversed(messages):
            m_tok = count_tokens(m.get("content", ""))
            if msg_tokens + m_tok > 2000 and recent_msgs:
                break
            recent_msgs.insert(0, m)
            msg_tokens += m_tok

        for m in recent_msgs:
            role = "USER" if m["role"] == "human" else "AGENT"
            parts.append(f"{role}: {m['content']}")

    return "\n".join(parts)
