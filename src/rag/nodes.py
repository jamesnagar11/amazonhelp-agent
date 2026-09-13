from __future__ import annotations
"""
All LangGraph node functions for the AmazonHelp CRAG pipeline.

Node list:
  get_intent → intent_evaluator → rewrite_query → retriever → eval_retriever
  → correct → brain_node
  → ambiguous → ambiguous_retriever → recompose_ambiguous_strips → brain_node
  → incorrect → human_node
  human_node → END
  brain_node → END
"""

from langchain_core.documents import Document

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field
from langsmith import traceable

from src.rag.state import GraphState
from src.rag.prompts import (
    GET_INTENT_PROMPT,
    REWRITE_QUERY_PROMPT,
    EVAL_RETRIEVER_PROMPT,
    STRIP_SCORE_PROMPT,
    INCORRECT_ESCALATION_PROMPT,
    AMBIGUOUS_REWRITE_PROMPT,
    AMBIGUOUS_ESCALATION_PROMPT,
    BRAIN_NODE_PROMPT,
    INTENT_ESCALATION_PROMPT,
    INTENT_LIST,
)
from src.rag.memory import build_context_string

logger = logging.getLogger(__name__)

STRIP_SCORE_THRESHOLD = 0.45
CORRECT_THRESHOLD = 0.8
AMBIGUOUS_LOW = 0.35
AMBIGUOUS_HIGH = 0.7


# ── Pydantic Output Schemas ──────────────────────────────────────────────────

class IntentOutput(BaseModel):
    intent: str = Field(description="One of the 25 exact AmazonHelp intents")
    intent_score: int = Field(description="Base severity score 1-10")

class RewriteQueryOutput(BaseModel):
    user_query_rewritten: str = Field(description="Enhanced query for vector search")
    fake_answer: str = Field(description="Short hypothetical answer for HyDE retrieval")

class DocScoreItem(BaseModel):
    doc_index: int = Field(description="0-based document index")
    score: float = Field(description="Relevance score 0.0 to 1.0")

class EvalRetrieverOutput(BaseModel):
    scores: list[DocScoreItem] = Field(default_factory=list, description="Scores for retrieved documents")

class EscalationOutput(BaseModel):
    escalation_reason: str = Field(description="Escalation summary and reasoning for human agent")

class AmbiguousRewriteOutput(BaseModel):
    refined_query: str = Field(description="Targeted query addressing missing context")
    refined_fake_answer: str = Field(description="Refined hypothetical answer")

class StripScoreItem(BaseModel):
    strip_index: int = Field(description="0-based strip index")
    score: float = Field(description="Relevance score 0.0 to 1.0")

class BrainNodeOutput(BaseModel):
    response: str = Field(description="Final customer support response")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json(text: str, fallback: Any = None) -> Any:
    """Robustly extract JSON from LLM output (handles markdown fences)."""
    if hasattr(text, "content"):
        text = text.content
    text = str(text).strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    # Find first {...} or [...] block
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except Exception:
        logger.warning("JSON parse failed for: %s…", text[:200])
        return fallback


def _docs_to_text(docs: list[Document]) -> str:
    out = []
    for i, doc in enumerate(docs):
        out.append(f"[Document {i}]\n{doc.page_content}")
    return "\n\n".join(out)


def _split_into_strips(text: str) -> list[str]:
    """Split document text into atomic strips (sentences/lines)."""
    # Split on newlines, then on sentence-ending punctuation
    raw = re.split(r"\n+|(?<=[.!?])\s+", text)
    strips = [s.strip() for s in raw if s.strip() and len(s.strip()) > 10]
    return strips


def _score_and_filter_strips(
    strips: list[str],
    query: str,
    intent: str,
    llm: Any,
    threshold: float = STRIP_SCORE_THRESHOLD,
) -> tuple[list[str], list[dict]]:
    """Score strips with LLM and return (kept_strips, all_scores_list)."""
    if not strips:
        return [], []

    strips_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(strips))
    prompt = STRIP_SCORE_PROMPT.format(
        query=query,
        intent=intent,
        strips_text=strips_text,
    )
    try:
        raw = llm.invoke(prompt)
        scores_list = _parse_json(raw, fallback=[])
        if not isinstance(scores_list, list):
            scores_list = []
    except Exception as e:
        logger.warning("Strip scoring failed: %s", e)
        scores_list = []

    # Build score lookup
    score_map: dict[int, float] = {}
    for item in scores_list:
        if isinstance(item, dict):
            idx = item.get("strip_index", item.get("index", -1))
            score = item.get("score", 0.0)
            if isinstance(idx, int) and 0 <= idx < len(strips):
                score_map[idx] = float(score)

    kept = []
    all_scores = []
    for i, strip in enumerate(strips):
        s = score_map.get(i, 0.0)
        all_scores.append({"strip_index": i, "score": s, "text": strip})
        if s >= threshold:
            kept.append(strip)

    return kept, all_scores


@traceable(name="detect_language", tags=["utility", "nlp"], metadata={"component": "language_detector"})
def detect_language(text: str) -> str:
    """Detect language of text (e.g. English, French, Spanish, German, Japanese)."""
    if not text or len(text.strip()) < 3:
        return "English"
    try:
        from langdetect import detect
        code = detect(text)
        lang_map = {
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "ja": "Japanese",
            "pt": "Portuguese",
            "it": "Italian",
            "hi": "Hindi",
            "zh-cn": "Chinese",
            "zh-tw": "Chinese",
        }
        return lang_map.get(code.lower(), code)
    except Exception:
        return "English"


# ── NODES ────────────────────────────────────────────────────────────────────

@traceable(name="get_intent", tags=["node", "intent"], metadata={"step": "1_intent_classification"})
def get_intent(state: GraphState) -> GraphState:
    """Classify user query into one of 25 AmazonHelp intents & detect language."""
    from src.utils.llm import get_chat_llm

    llm = get_chat_llm()
    query = state.get("query", "")
    detected_language = detect_language(query)

    prompt = GET_INTENT_PROMPT.format(
        intent_list=INTENT_LIST,
        query=query,
    )
    try:
        if hasattr(llm, "with_structured_output"):
            try:
                s_llm = llm.with_structured_output(IntentOutput)
                res = s_llm.invoke(prompt)
                if isinstance(res, IntentOutput):
                    intent = res.intent
                    intent_score = res.intent_score
                elif isinstance(res, dict):
                    intent = res.get("intent", "General/Other")
                    intent_score = int(res.get("intent_score", 3))
                else:
                    parsed = _parse_json(res, fallback={})
                    intent = parsed.get("intent", "General/Other")
                    intent_score = int(parsed.get("intent_score", 3))
            except Exception:
                raw = llm.invoke(prompt)
                parsed = _parse_json(raw, fallback={})
                intent = parsed.get("intent", "General/Other")
                intent_score = int(parsed.get("intent_score", 3))
        else:
            raw = llm.invoke(prompt)
            parsed = _parse_json(raw, fallback={})
            intent = parsed.get("intent", "General/Other")
            intent_score = int(parsed.get("intent_score", 3))
    except Exception as e:
        logger.warning("get_intent failed: %s", e)
        intent = "General/Other"
        intent_score = 3

    logger.info("Intent: %s | Score: %d | Lang: %s", intent, intent_score, detected_language)
    return {**state, "intent": intent, "intent_score": intent_score, "detected_language": detected_language}


@traceable(name="intent_evaluator", tags=["node", "evaluator"], metadata={"step": "2_intent_evaluation"})
def intent_evaluator(state: GraphState) -> GraphState:
    """
    Evaluate intent score + apply escalation formula.
    Decides to escalate to human or proceed to rewrite_query.
    formula: final_score = base_severity + min(iteration_count × 1, 2) + (2 if context_complete_but_unresolved else 0)
    """
    from src.utils.llm import get_judge_llm

    intent = state.get("intent", "")
    intent_score = state.get("intent_score", 3)
    iteration_count = state.get("iteration_count", 0)

    # Immediate escalation intents
    IMMEDIATE_ESCALATE = {
        "Account Security/Hacked",
        "Unauthorized/Incorrect Charge",
        "Customer Service Complaint (Escalation)",
    }

    # Check immediate escalation
    immediate = intent in IMMEDIATE_ESCALATE and intent_score >= 8

    # Formula: final_score = base_severity + min(iteration_count, 2) + (2 if unresolved)
    context_unresolved = iteration_count >= 2  # treat repeated queries as unresolved
    final_score = intent_score + min(iteration_count, 2) + (2 if context_unresolved else 0)

    should_escalate = immediate or final_score >= 11

    logger.info("final_score=%d | escalate=%s", final_score, should_escalate)

    if should_escalate:
        llm = get_judge_llm()
        prompt = INTENT_ESCALATION_PROMPT.format(
            query=state.get("query", ""),
            intent=intent,
            intent_score=intent_score,
            final_score=final_score,
            iteration_count=iteration_count,
        )
        try:
            if hasattr(llm, "with_structured_output"):
                try:
                    s_llm = llm.with_structured_output(EscalationOutput)
                    res = s_llm.invoke(prompt)
                    if isinstance(res, EscalationOutput):
                        escalation_reason = res.escalation_reason
                    else:
                        parsed = _parse_json(res, fallback={})
                        escalation_reason = parsed.get("escalation_reason", "High-severity issue requiring human intervention.")
                except Exception:
                    raw = llm.invoke(prompt)
                    parsed = _parse_json(raw, fallback={})
                    escalation_reason = parsed.get("escalation_reason", "High-severity issue requiring human intervention.")
            else:
                raw = llm.invoke(prompt)
                parsed = _parse_json(raw, fallback={})
                escalation_reason = parsed.get("escalation_reason", "High-severity issue requiring human intervention.")
        except Exception as e:
            logger.warning("Escalation reason generation failed: %s", e)
            escalation_reason = f"Escalated due to high intent score ({final_score}) for intent: {intent}"

        return {
            **state,
            "final_score": final_score,
            "human_escalated": True,
            "escalation_reason": escalation_reason,
        }

    return {**state, "final_score": final_score, "human_escalated": False}


@traceable(name="rewrite_query", tags=["node", "rewrite"], metadata={"step": "3_hyde_rewrite"})
def rewrite_query(state: GraphState) -> GraphState:
    """Enhance query + generate HyDE fake answer using DeepSeek."""
    from src.utils.llm import get_judge_llm

    llm = get_judge_llm()
    prompt = REWRITE_QUERY_PROMPT.format(
        query=state.get("query", ""),
        intent=state.get("intent", ""),
        intent_score=state.get("intent_score", 3),
    )
    try:
        if hasattr(llm, "with_structured_output"):
            try:
                s_llm = llm.with_structured_output(RewriteQueryOutput)
                res = s_llm.invoke(prompt)
                if isinstance(res, RewriteQueryOutput):
                    rewritten = res.user_query_rewritten
                    fake_answer = res.fake_answer
                else:
                    parsed = _parse_json(res, fallback={})
                    rewritten = parsed.get("user_query_rewritten", state.get("query", ""))
                    fake_answer = parsed.get("fake_answer", "")
            except Exception:
                raw = llm.invoke(prompt)
                parsed = _parse_json(raw, fallback={})
                rewritten = parsed.get("user_query_rewritten", state.get("query", ""))
                fake_answer = parsed.get("fake_answer", "")
        else:
            raw = llm.invoke(prompt)
            parsed = _parse_json(raw, fallback={})
            rewritten = parsed.get("user_query_rewritten", state.get("query", ""))
            fake_answer = parsed.get("fake_answer", "")
    except Exception as e:
        logger.warning("rewrite_query failed: %s", e)
        rewritten = state.get("query", "")
        fake_answer = ""

    logger.info("Rewritten query: %s…", rewritten[:100])
    return {**state, "user_query_rewritten": rewritten, "fake_answer": fake_answer}


@traceable(name="retriever", tags=["node", "retrieval"], metadata={"step": "4_qdrant_retrieval"})
def retriever(state: GraphState) -> GraphState:
    """Retrieve top-5 documents from Qdrant using rewritten query + fake answer."""
    from src.utils.vector_store import get_retriever, collection_exists

    if not collection_exists():
        logger.warning("Qdrant collection is empty! Run ingestion first.")
        return {**state, "documents": []}

    ret = get_retriever(k=5)
    combined_query = f"{state.get('user_query_rewritten', state.get('query', ''))}\n{state.get('fake_answer', '')}"

    try:
        docs = ret.invoke(combined_query)
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        docs = []

    logger.info("Retrieved %d documents", len(docs))
    return {**state, "documents": docs}


@traceable(name="eval_retriever", tags=["node", "evaluator"], metadata={"step": "5_doc_scoring"})
def eval_retriever(state: GraphState) -> GraphState:
    """
    Score each retrieved document 0.0–1.0 using DeepSeek.
    Classify into correct (>0.8), ambiguous (0.35–0.8), incorrect (<0.35).
    """
    from src.utils.llm import get_judge_llm

    docs = state.get("documents", [])
    if not docs:
        return {
            **state,
            "evaluated_docs": [],
            "correct_docs": [],
            "ambiguous_docs": [],
        }

    llm = get_judge_llm()
    docs_text = _docs_to_text(docs)

    prompt = EVAL_RETRIEVER_PROMPT.format(
        query=state.get("query", ""),
        user_query_rewritten=state.get("user_query_rewritten", ""),
        intent=state.get("intent", ""),
        num_docs=len(docs),
        documents_text=docs_text,
    )
    try:
        if hasattr(llm, "with_structured_output"):
            try:
                s_llm = llm.with_structured_output(EvalRetrieverOutput)
                res = s_llm.invoke(prompt)
                if isinstance(res, EvalRetrieverOutput):
                    scores_list = [{"doc_index": item.doc_index, "score": item.score} for item in res.scores]
                else:
                    scores_list = _parse_json(res, fallback=[])
            except Exception:
                raw = llm.invoke(prompt)
                scores_list = _parse_json(raw, fallback=[])
        else:
            raw = llm.invoke(prompt)
            scores_list = _parse_json(raw, fallback=[])
    except Exception as e:
        logger.warning("eval_retriever LLM failed: %s", e)
        scores_list = []

    # Build evaluated_docs
    score_map = {item.get("doc_index", -1): float(item.get("score", 0.0))
                 for item in scores_list if isinstance(item, dict)}

    evaluated = []
    for i, doc in enumerate(docs):
        score = score_map.get(i, 0.0)
        evaluated.append({"doc": doc, "score": score})
        logger.info("Doc %d score: %.2f", i, score)

    # Classify cleanly into mutually exclusive buckets
    correct_docs = [e["doc"] for e in evaluated if e["score"] >= AMBIGUOUS_HIGH]
    ambiguous_docs = [
        e["doc"] for e in evaluated if AMBIGUOUS_LOW <= e["score"] < AMBIGUOUS_HIGH
    ]

    return {
        **state,
        "evaluated_docs": evaluated,
        "correct_docs": correct_docs,
        "ambiguous_docs": ambiguous_docs,
    }


@traceable(name="correct", tags=["node", "strip_filtering"], metadata={"step": "6_correct_path"})
def correct(state: GraphState) -> GraphState:
    """
    Decompose correct_docs into strips, score strips, filter (<0.45), recompose.
    Output: refined_context string.
    """
    from src.utils.llm import get_chat_llm

    llm = get_chat_llm()
    correct_docs = state.get("correct_docs", [])
    query = state.get("query", "")
    intent = state.get("intent", "")

    # Combine all correct doc texts
    combined_text = "\n\n".join(doc.page_content for doc in correct_docs)
    strips = _split_into_strips(combined_text)

    kept_strips, _ = _score_and_filter_strips(strips, query, intent, llm)

    if not kept_strips:
        # Fall back to raw text if nothing survives
        refined_context = combined_text[:2000]
    else:
        refined_context = "\n".join(kept_strips)

    logger.info("correct node: %d strips → %d kept", len(strips), len(kept_strips))
    return {**state, "refined_context": refined_context}


@traceable(name="incorrect", tags=["node", "escalation"], metadata={"step": "6_incorrect_path"})
def incorrect(state: GraphState) -> GraphState:
    """
    All docs scored < 0.35. Generate escalation reason and route to human_node.
    """
    from src.utils.llm import get_chat_llm

    llm = get_chat_llm()
    evaluated = state.get("evaluated_docs", [])
    doc_scores = ", ".join(f"{e['score']:.2f}" for e in evaluated)

    prompt = INCORRECT_ESCALATION_PROMPT.format(
        query=state.get("query", ""),
        intent=state.get("intent", ""),
        intent_score=state.get("intent_score", 3),
        iteration_count=state.get("iteration_count", 0),
        doc_scores=doc_scores,
    )
    try:
        if hasattr(llm, "with_structured_output"):
            try:
                s_llm = llm.with_structured_output(EscalationOutput)
                res = s_llm.invoke(prompt)
                if isinstance(res, EscalationOutput):
                    escalation_reason = res.escalation_reason
                else:
                    parsed = _parse_json(res, fallback={})
                    escalation_reason = parsed.get("escalation_reason", "No relevant information found in knowledge base.")
            except Exception:
                raw = llm.invoke(prompt)
                parsed = _parse_json(raw, fallback={})
                escalation_reason = parsed.get("escalation_reason", "No relevant information found in knowledge base.")
        else:
            raw = llm.invoke(prompt)
            parsed = _parse_json(raw, fallback={})
            escalation_reason = parsed.get("escalation_reason", "No relevant information found in knowledge base.")
    except Exception as e:
        logger.warning("incorrect node LLM failed: %s", e)
        escalation_reason = "Unable to find relevant information. Routing to human agent."

    return {**state, "escalation_reason": escalation_reason, "human_escalated": True}


@traceable(name="ambiguous", tags=["node", "rewrite"], metadata={"step": "6_ambiguous_path"})
def ambiguous(state: GraphState) -> GraphState:
    """
    Re-write query focusing on the weak/ambiguous sections to improve retrieval.
    """
    from src.utils.llm import get_chat_llm

    llm = get_chat_llm()
    evaluated = state.get("evaluated_docs", [])
    ambiguous_docs = state.get("ambiguous_docs", [])

    # Collect excerpts from low-scoring docs for context
    low_score_excerpts = "\n\n".join(
        f"[score={e['score']:.2f}] {e['doc'].page_content[:300]}"
        for e in evaluated if e["score"] < AMBIGUOUS_HIGH
    )

    prompt = AMBIGUOUS_REWRITE_PROMPT.format(
        query=state.get("query", ""),
        user_query_rewritten=state.get("user_query_rewritten", ""),
        intent=state.get("intent", ""),
        intent_score=state.get("intent_score", 3),
        low_score_excerpts=low_score_excerpts or "No low-scoring excerpts available.",
    )
    try:
        if hasattr(llm, "with_structured_output"):
            try:
                s_llm = llm.with_structured_output(AmbiguousRewriteOutput)
                res = s_llm.invoke(prompt)
                if isinstance(res, AmbiguousRewriteOutput):
                    refined_query = res.refined_query
                    refined_fake = res.refined_fake_answer
                else:
                    parsed = _parse_json(res, fallback={})
                    refined_query = parsed.get("refined_query", state.get("user_query_rewritten", ""))
                    refined_fake = parsed.get("refined_fake_answer", state.get("fake_answer", ""))
            except Exception:
                raw = llm.invoke(prompt)
                parsed = _parse_json(raw, fallback={})
                refined_query = parsed.get("refined_query", state.get("user_query_rewritten", ""))
                refined_fake = parsed.get("refined_fake_answer", state.get("fake_answer", ""))
        else:
            raw = llm.invoke(prompt)
            parsed = _parse_json(raw, fallback={})
            refined_query = parsed.get("refined_query", state.get("user_query_rewritten", ""))
            refined_fake = parsed.get("refined_fake_answer", state.get("fake_answer", ""))
    except Exception as e:
        logger.warning("ambiguous node failed: %s", e)
        refined_query = state.get("user_query_rewritten", state.get("query", ""))
        refined_fake = state.get("fake_answer", "")

    logger.info("Ambiguous refined query: %s…", refined_query[:100])
    return {**state, "user_query_rewritten": refined_query, "fake_answer": refined_fake}


@traceable(name="ambiguous_retriever", tags=["node", "retrieval"], metadata={"step": "7_ambiguous_retrieval"})
def ambiguous_retriever(state: GraphState) -> GraphState:
    """Retrieve top-4 documents using refined ambiguous query."""
    from src.utils.vector_store import get_retriever, collection_exists

    if not collection_exists():
        return {**state, "documents": []}

    ret = get_retriever(k=4)
    combined_query = f"{state.get('user_query_rewritten', '')}\n{state.get('fake_answer', '')}"

    try:
        docs = ret.invoke(combined_query)
    except Exception as e:
        logger.error("ambiguous_retriever failed: %s", e)
        docs = []

    logger.info("ambiguous_retriever: retrieved %d documents", len(docs))
    return {**state, "documents": docs}


@traceable(name="recompose_ambiguous_strips", tags=["node", "strip_filtering"], metadata={"step": "8_ambiguous_recompose"})
def recompose_ambiguous_strips(state: GraphState) -> GraphState:
    """
    Score strips from 4 ambiguous docs, filter, recompose.
    If < 10% strips above 0.4 → escalate to human.
    """
    from src.utils.llm import get_chat_llm

    llm = get_chat_llm()
    docs = state.get("documents", [])
    query = state.get("query", "")
    intent = state.get("intent", "")

    combined_text = "\n\n".join(doc.page_content for doc in docs)
    strips = _split_into_strips(combined_text)

    kept_strips, all_scores = _score_and_filter_strips(strips, query, intent, llm, threshold=0.45)

    # Check if less than 10% of strips are above 0.4
    above_threshold = sum(1 for s in all_scores if s["score"] > 0.4)
    total = len(all_scores)
    threshold_pct = (above_threshold / total) if total > 0 else 0

    logger.info(
        "recompose_ambiguous_strips: %d/%d strips above 0.4 (%.0f%%)",
        above_threshold, total, threshold_pct * 100
    )

    if threshold_pct < 0.10 or not kept_strips:
        # Escalate to human
        strip_scores_summary = f"{above_threshold}/{total} strips above 0.4 threshold"
        prompt = AMBIGUOUS_ESCALATION_PROMPT.format(
            query=query,
            intent=intent,
            intent_score=state.get("intent_score", 3),
            strip_scores_summary=strip_scores_summary,
        )
        try:
            if hasattr(llm, "with_structured_output"):
                try:
                    s_llm = llm.with_structured_output(EscalationOutput)
                    res = s_llm.invoke(prompt)
                    if isinstance(res, EscalationOutput):
                        escalation_reason = res.escalation_reason
                    else:
                        parsed = _parse_json(res, fallback={})
                        escalation_reason = parsed.get("escalation_reason", "Insufficient relevant content after two-stage retrieval.")
                except Exception:
                    raw = llm.invoke(prompt)
                    parsed = _parse_json(raw, fallback={})
                    escalation_reason = parsed.get("escalation_reason", "Insufficient relevant content after two-stage retrieval.")
            else:
                raw = llm.invoke(prompt)
                parsed = _parse_json(raw, fallback={})
                escalation_reason = parsed.get("escalation_reason", "Insufficient relevant content after two-stage retrieval.")
        except Exception as e:
            logger.warning("recompose escalation prompt failed: %s", e)
            escalation_reason = "Insufficient relevant content found after two-stage retrieval. Human agent required."

        return {**state, "escalation_reason": escalation_reason, "human_escalated": True}

    refined_context = "\n".join(kept_strips)
    return {**state, "refined_context": refined_context, "human_escalated": False}


@traceable(name="brain_node", tags=["node", "generation"], metadata={"step": "9_brain_llm_response"})
def brain_node(state: GraphState) -> GraphState:
    """
    Final answer generation using Qwen3 with full context:
    refined_context + conversation history + intent info.
    """
    from src.utils.llm import get_chat_llm

    llm = get_chat_llm(max_tokens=512)

    conversation_history = build_context_string(
        state.get("summary_list", []),
        state.get("messages", []),
    )

    detected_language = state.get("detected_language", "English")

    prompt = BRAIN_NODE_PROMPT.format(
        query=state.get("query", ""),
        detected_language=detected_language,
        intent=state.get("intent", ""),
        intent_score=state.get("intent_score", 3),
        iteration_count=state.get("iteration_count", 0),
        conversation_history=conversation_history or "No previous conversation.",
        refined_context=state.get("refined_context", "No relevant context retrieved."),
    )
    try:
        if hasattr(llm, "with_structured_output"):
            try:
                s_llm = llm.with_structured_output(BrainNodeOutput)
                res = s_llm.invoke(prompt)
                if isinstance(res, BrainNodeOutput):
                    response = res.response
                elif isinstance(res, dict):
                    response = res.get("response", "")
                else:
                    parsed = _parse_json(res, fallback={})
                    response = parsed.get("response", "")
                if not response:
                    response = str(res.content if hasattr(res, "content") else res).strip()
            except Exception:
                raw = llm.invoke(prompt)
                parsed = _parse_json(raw, fallback={})
                response = parsed.get("response", "")
                if not response:
                    response = str(raw.content if hasattr(raw, "content") else raw).strip()
        else:
            raw = llm.invoke(prompt)
            parsed = _parse_json(raw, fallback={})
            response = parsed.get("response", "")
            if not response:
                response = str(raw.content if hasattr(raw, "content") else raw).strip()
    except Exception as e:
        logger.error("brain_node failed: %s", e)
        response = (
            "I'm sorry, I encountered an issue generating a response. "
            "Our @AmazonHelp support team is reviewing your request to assist you."
        )

    logger.info("brain_node response: %s…", response[:100])
    return {**state, "response": response, "human_escalated": False}


@traceable(name="human_node", tags=["node", "escalation"], metadata={"step": "9_human_escalation"})
def human_node(state: GraphState) -> GraphState:
    """
    Format the escalation message for the user.
    Called when human intervention is required.
    """
    intent = state.get("intent", "Unknown")
    escalation_reason = state.get("escalation_reason", "Your request requires human assistance.")
    query = state.get("query", "")
    intent_score = state.get("intent_score", 3)

    response = (
        f"🚨 **Your request has been escalated to a senior AmazonHelp customer support specialist.**\n\n"
        f"**Issue Category:** {intent}\n\n"
        f"**Escalation Details:**\n{escalation_reason}\n\n"
        f"---\n\n"
        f"**Status & Next Steps:**\n"
        f"As an automated @AmazonHelp support agent, I have logged your case and transferred it directly to a human support specialist. "
        f"A support team member will reach out to you within 24 hours "
        f"{'(typically within 1–2 hours for urgent issues)' if intent_score >= 8 else ''} to resolve your inquiry:\n\n"
        f"> *\"{query}\"*\n\n"
        f"Thank you for your patience while our specialist team reviews your request. 🙏"
    )

    return {**state, "response": response, "human_escalated": True}


# ── Conditional edge functions ────────────────────────────────────────────────

@traceable(name="route_after_intent_evaluator", tags=["router", "intent"], metadata={"component": "conditional_edge"})
def route_after_intent_evaluator(state: GraphState) -> str:
    """Route to human_node or rewrite_query."""
    if state.get("human_escalated", False):
        return "human_node"
    return "rewrite_query"


@traceable(name="route_after_eval_retriever", tags=["router", "retriever"], metadata={"component": "conditional_edge"})
def route_after_eval_retriever(state: GraphState) -> str:
    """
    Route based on document evaluation scores:
    - correct → correct
    - all low → incorrect
    - mixed → ambiguous
    """
    evaluated = state.get("evaluated_docs", [])
    if not evaluated:
        return "incorrect"

    scores = [e["score"] for e in evaluated]
    max_score = max(scores)
    min_score = min(scores)

    # Any doc above 0.8 → correct path
    if max_score > CORRECT_THRESHOLD:
        return "correct"
    # All docs below 0.35 → incorrect
    if max_score < AMBIGUOUS_LOW:
        return "incorrect"
    # Mixed → ambiguous
    return "ambiguous"


@traceable(name="route_after_recompose_ambiguous", tags=["router", "ambiguous"], metadata={"component": "conditional_edge"})
def route_after_recompose_ambiguous(state: GraphState) -> str:
    """Route to human_node or brain_node after ambiguous recompose."""
    if state.get("human_escalated", False):
        return "human_node"
    return "brain_node"
