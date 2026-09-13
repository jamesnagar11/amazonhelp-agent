"""
LangGraph graph assembly and compilation for the AmazonHelp CRAG pipeline.

Graph flow:
  START → get_intent → intent_evaluator
        ↘ (human_escalated) → human_node → END
        ↘ (else)            → rewrite_query → retriever → eval_retriever
                                ↘ correct           → correct → brain_node → END
                                ↘ ambiguous         → ambiguous → ambiguous_retriever
                                                      → recompose_ambiguous_strips
                                                          ↘ (escalate) → human_node → END
                                                          ↘ (else)     → brain_node → END
                                ↘ incorrect         → incorrect → human_node → END
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable

from src.rag.state import GraphState
from src.rag.nodes import (
    get_intent,
    intent_evaluator,
    rewrite_query,
    retriever,
    eval_retriever,
    correct,
    incorrect,
    ambiguous,
    ambiguous_retriever,
    recompose_ambiguous_strips,
    brain_node,
    human_node,
    route_after_intent_evaluator,
    route_after_eval_retriever,
    route_after_recompose_ambiguous,
)

logger = logging.getLogger(__name__)

SQLITE_PATH = os.getenv("SQLITE_PATH", "./checkpoints/chat_checkpoints.db")


@traceable(name="build_graph", tags=["graph", "builder"], metadata={"component": "graph_builder"})
def build_graph() -> StateGraph:
    """Build and return the uncompiled StateGraph."""
    builder = StateGraph(GraphState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    builder.add_node("get_intent", get_intent)
    builder.add_node("intent_evaluator", intent_evaluator)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retriever", retriever)
    builder.add_node("eval_retriever", eval_retriever)
    builder.add_node("correct", correct)
    builder.add_node("incorrect", incorrect)
    builder.add_node("ambiguous", ambiguous)
    builder.add_node("ambiguous_retriever", ambiguous_retriever)
    builder.add_node("recompose_ambiguous_strips", recompose_ambiguous_strips)
    builder.add_node("brain_node", brain_node)
    builder.add_node("human_node", human_node)

    # ── Add edges ──────────────────────────────────────────────────────────
    builder.add_edge(START, "get_intent")
    builder.add_edge("get_intent", "intent_evaluator")

    # Conditional: intent_evaluator → rewrite_query OR human_node
    builder.add_conditional_edges(
        "intent_evaluator",
        route_after_intent_evaluator,
        {
            "rewrite_query": "rewrite_query",
            "human_node": "human_node",
        },
    )

    builder.add_edge("rewrite_query", "retriever")
    builder.add_edge("retriever", "eval_retriever")

    # Conditional: eval_retriever → correct / ambiguous / incorrect
    builder.add_conditional_edges(
        "eval_retriever",
        route_after_eval_retriever,
        {
            "correct": "correct",
            "ambiguous": "ambiguous",
            "incorrect": "incorrect",
        },
    )

    # Correct path
    builder.add_edge("correct", "brain_node")

    # Incorrect path
    builder.add_edge("incorrect", "human_node")

    # Ambiguous path
    builder.add_edge("ambiguous", "ambiguous_retriever")
    builder.add_edge("ambiguous_retriever", "recompose_ambiguous_strips")

    # Conditional: recompose_ambiguous_strips → brain_node OR human_node
    builder.add_conditional_edges(
        "recompose_ambiguous_strips",
        route_after_recompose_ambiguous,
        {
            "brain_node": "brain_node",
            "human_node": "human_node",
        },
    )

    # Terminal edges
    builder.add_edge("brain_node", END)
    builder.add_edge("human_node", END)

    return builder


def get_compiled_graph():
    """Compile the graph with SqliteSaver checkpointer."""
    import sqlite3
    # Ensure checkpoint directory exists
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)

    builder = build_graph()
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Graph compiled with SqliteSaver at: %s", SQLITE_PATH)
    return graph


# ── Singleton graph (cached after first call) ─────────────────────────────────
_compiled_graph = None


def get_graph():
    """Return the singleton compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = get_compiled_graph()
    return _compiled_graph


@traceable(name="invoke_graph", tags=["pipeline", "execution"], metadata={"component": "pipeline_executor"})
def invoke_graph(
    query: str,
    thread_id: str,
    messages: list[dict] | None = None,
    summary_list: list[str] | None = None,
    iteration_count: int = 0,
) -> dict:
    """
    Invoke the graph for a single user query.

    Args:
        query: The user's message
        thread_id: Unique chat session identifier
        messages: Recent conversation history (last MAX_MESSAGES)
        summary_list: List of past summaries
        iteration_count: Number of previous turns in this thread

    Returns:
        Final graph state dict
    """
    from src.rag.memory import trim_messages, save_thread_state

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: GraphState = {
        "query": query,
        "thread_id": thread_id,
        "messages": messages or [],
        "summary_list": summary_list or [],
        "iteration_count": iteration_count,
        "intent": "",
        "intent_score": 0,
        "final_score": 0,
        "user_query_rewritten": "",
        "fake_answer": "",
        "documents": [],
        "evaluated_docs": [],
        "correct_docs": [],
        "ambiguous_docs": [],
        "refined_context": "",
        "response": "",
        "escalation_reason": "",
        "human_escalated": False,
    }

    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error("Graph invocation failed: %s", e)
        result = {
            **initial_state,
            "response": (
                "I'm sorry, I encountered a technical issue. "
                "Please contact Amazon support at amazon.com/help."
            ),
            "human_escalated": False,
        }

    # ── Update memory after successful run ────────────────────────────────
    response_text = result.get("response", "")
    updated_messages = list(messages or []) + [
        {"role": "human", "content": query},
        {"role": "ai", "content": response_text},
    ]

    # Trim with sliding window + summarise overflow
    trimmed_messages, updated_summaries = trim_messages(
        updated_messages,
        list(summary_list or []),
    )

    new_iteration = iteration_count + 1

    # Persist to metadata DB
    save_thread_state(
        thread_id=thread_id,
        messages=trimmed_messages,
        summary_list=updated_summaries,
        iteration_count=new_iteration,
    )

    result["messages"] = trimmed_messages
    result["summary_list"] = updated_summaries
    result["iteration_count"] = new_iteration

    return result


@traceable(name="stream_graph", tags=["pipeline", "streaming"], metadata={"component": "pipeline_streamer"})
def stream_graph(
    query: str,
    thread_id: str,
    messages: list[dict] | None = None,
    summary_list: list[str] | None = None,
    iteration_count: int = 0,
):
    """
    Stream graph node outputs for real-time UI updates.
    Yields (node_name, state_update) tuples.
    """
    from src.rag.memory import trim_messages, save_thread_state

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: GraphState = {
        "query": query,
        "thread_id": thread_id,
        "messages": messages or [],
        "summary_list": summary_list or [],
        "iteration_count": iteration_count,
        "intent": "",
        "intent_score": 0,
        "final_score": 0,
        "user_query_rewritten": "",
        "fake_answer": "",
        "documents": [],
        "evaluated_docs": [],
        "correct_docs": [],
        "ambiguous_docs": [],
        "refined_context": "",
        "response": "",
        "escalation_reason": "",
        "human_escalated": False,
    }

    final_result = initial_state.copy()

    for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
        if isinstance(chunk, dict):
            for node_name, state_update in chunk.items():
                if not isinstance(state_update, dict):
                    continue
                final_result.update(state_update)
                yield node_name, state_update

                # Incremental memory save when response node completes
                if node_name in ("brain_node", "human_node") and "response" in state_update and state_update["response"]:
                    response_text = state_update["response"]
                    updated_messages = list(messages or []) + [
                        {"role": "human", "content": query},
                        {"role": "ai", "content": response_text},
                    ]
                    trimmed_messages, updated_summaries = trim_messages(
                        updated_messages,
                        list(summary_list or []),
                    )
                    save_thread_state(
                        thread_id=thread_id,
                        messages=trimmed_messages,
                        summary_list=updated_summaries,
                        iteration_count=iteration_count + 1,
                    )
