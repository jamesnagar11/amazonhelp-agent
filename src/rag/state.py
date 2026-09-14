"""
GraphState TypedDict for the AmazonHelp CRAG LangGraph pipeline.
All nodes read from and write back to this shared state.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from langchain_core.documents import Document


def _merge_list(a: list, b: list) -> list:
    """Reducer: replace list state entirely with the new value."""
    return b if b is not None else a


class GraphState(TypedDict, total=False):
    # ── User input ─────────────────────────────────────────────────────────
    query: str                          # raw user message
    thread_id: str                      # unique chat session ID
    detected_language: str              # ISO code (e.g. 'en', 'fr', 'es', 'de', 'ja')

    # ── Intent classification ───────────────────────────────────────────────
    intent: str                         # one of 25 AmazonHelp intents
    intent_score: int                   # base severity 1–10
    final_score: int                    # after formula

    # ── Iteration tracking ─────────────────────────────────────────────────
    iteration_count: int                # total turns in this thread

    # ── Rewritten query ────────────────────────────────────────────────────
    user_query_rewritten: str           # enhanced query
    fake_answer: str                    # hypothetical answer for HyDE retrieval

    # ── Retrieved documents ────────────────────────────────────────────────
    documents: list[Document]           # top-5 raw retrieved docs
    evaluated_docs: list[dict]          # [{"doc": Document, "score": float}]
    correct_docs: list[Document]        # docs with score > 0.8
    ambiguous_docs: list[Document]      # docs 0.35 ≤ score ≤ 0.8

    # ── Processed context ──────────────────────────────────────────────────
    refined_context: str                # strip-filtered single context string

    # ── Query rejection (non-Amazon / gibberish / off-topic) ───────────────
    rejected: bool                      # True if query was rejected before pipeline
    reject_reason: str | None           # LLM-provided reason, or None if not rejected

    # ── Final output ───────────────────────────────────────────────────────
    response: str                       # final answer to user
    escalation_reason: str             # why a human is needed
    human_escalated: bool               # True if human_node was invoked

    # ── Memory & chat history ──────────────────────────────────────────────
    messages: list[dict]                # [{"role": "human"|"ai", "content": str}]
    summary_list: list[str]            # rolling summaries of old messages
    token_count: int                   # approximate token count of messages
