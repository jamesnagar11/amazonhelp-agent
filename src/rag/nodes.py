from __future__ import annotations
"""
All LangGraph node functions for the AmazonHelp CRAG pipeline.

Node list:
  get_intent → (rejected?) → END
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
from langgraph.graph import END

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

STRIP_SCORE_THRESHOLD = 0.3
CORRECT_THRESHOLD = 0.8
AMBIGUOUS_LOW = 0.35
AMBIGUOUS_HIGH = 0.7


# ── Pydantic Output Schemas ──────────────────────────────────────────────────

class IntentOutput(BaseModel):
    intent: str = Field(description="One of the 25 exact AmazonHelp intents")
    intent_score: int = Field(description="Base severity score 1-10")
    should_reject: bool = Field(default=False, description="True if query is off-topic, gibberish, or unrelated to Amazon")
    reject_reason: str | None = Field(default=None, description="LLM-provided rejection reason, or null if not rejected")

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
    """Robustly extract JSON from LLM output (handles reasoning <think> tags, markdown fences & truncated outputs)."""
    if hasattr(text, "content"):
        text = text.content
    raw_str = str(text).strip()
    if not raw_str:
        return fallback

    # Strip <think>...</think> reasoning blocks
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_str, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # 1. Try direct json.loads
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Match exact JSON block
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. Attempt repair for truncated JSON object
    obj_match = re.search(r"\{[\s\S]*", cleaned)
    if obj_match:
        snippet = obj_match.group(0)
        fixed = snippet
        if fixed.count('"') % 2 != 0:
            fixed += '"'
        if not fixed.rstrip().endswith("}"):
            fixed += "}"
        try:
            return json.loads(fixed)
        except Exception:
            pass

    # 4. Attempt repair for truncated JSON array
    arr_match = re.search(r"\[[\s\S]*", cleaned)
    if arr_match:
        snippet = arr_match.group(0)
        fixed = snippet
        if fixed.count('"') % 2 != 0:
            fixed += '"'
        if fixed.count('{') > fixed.count('}'):
            fixed += '}'
        if not fixed.rstrip().endswith("]"):
            fixed += "]"
        try:
            return json.loads(fixed)
        except Exception:
            pass

    # 5. Regex extraction fallbacks for structured output schemas
    # user_query_rewritten & fake_answer
    rewritten_m = re.search(r'"user_query_rewritten"\s*:\s*"([^"]*)', cleaned)
    fake_m = re.search(r'"fake_answer"\s*:\s*"([^"]*)', cleaned)
    if rewritten_m or fake_m:
        return {
            "user_query_rewritten": rewritten_m.group(1) if rewritten_m else "",
            "fake_answer": fake_m.group(1) if fake_m else "",
        }

    # intent & intent_score
    intent_m = re.search(r'"intent"\s*:\s*"([^"]*)"', cleaned)
    score_m = re.search(r'"intent_score"\s*:\s*(\d+)', cleaned)
    if intent_m or score_m:
        return {
            "intent": intent_m.group(1) if intent_m else "General/Other",
            "intent_score": int(score_m.group(1)) if score_m else 3,
        }

    # doc scores list
    doc_scores = []
    for d_match in re.finditer(r'\{\s*"doc_index"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*([0-9.]+)\s*\}', cleaned):
        doc_scores.append({"doc_index": int(d_match.group(1)), "score": float(d_match.group(2))})
    if doc_scores:
        return doc_scores

    logger.warning("JSON parse failed for: %s…", raw_str[:200])
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
        if hasattr(llm, "invoke"):
            raw = llm.invoke(str(prompt))
        else:
            raw = str(llm(str(prompt)))
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


def _safe_structured_invoke(llm: Any, prompt: str, schema: Any = None) -> Any:
    """
    Invoke LLM safely. Uses structured output if supported (OpenAI/Gemini/Groq),
    or cleanly invokes prompt + JSON parsing for HuggingFace/Inference API endpoints.
    Raises on unrecoverable errors instead of silently returning empty/broken results.
    """
    llm_type_name = type(llm).__name__

    if schema is not None and hasattr(llm, "with_structured_output") and "HuggingFace" not in llm_type_name:
        try:
            s_llm = llm.with_structured_output(schema)
            res = s_llm.invoke(prompt)
            if isinstance(res, schema):
                return res
            if isinstance(res, dict):
                return res
            parsed = _parse_json(res, fallback=None)
            if parsed is not None:
                return parsed
        except NotImplementedError:
            logger.debug("with_structured_output not supported by %s, falling back to raw invoke", llm_type_name)
        except Exception as e:
            logger.warning("with_structured_output failed (%s): %s — falling back to raw invoke", llm_type_name, e)

    # Raw invocation (HuggingFace or fallback path)
    try:
        raw = llm.invoke(prompt)
    except Exception as e:
        logger.error("LLM raw invoke failed (%s): %s", llm_type_name, e)
        raise

    if schema is not None:
        parsed = _parse_json(raw, fallback=None)
        if parsed is not None:
            return parsed
        # Last resort: return the raw AIMessage content as a dict so callers
        # can extract text from it (brain_node will handle this via the `else` branch)
        raw_content = raw.content if hasattr(raw, "content") else str(raw)
        logger.warning("_safe_structured_invoke: JSON parse failed, returning raw content wrapper")
        return {"_raw": raw_content}
    return raw


# ── NODES ────────────────────────────────────────────────────────────────────

@traceable(name="get_intent", tags=["node", "intent"], metadata={"step": "1_intent_classification"})
def get_intent(state: GraphState) -> GraphState:
    """Classify user query into one of 25 AmazonHelp intents & detect language.
    Also detects off-topic / gibberish queries and marks them for rejection.
    """
    from src.utils.llm import get_chat_llm

    llm = get_chat_llm()
    query = state.get("query", "")
    detected_language = detect_language(query)

    prompt = GET_INTENT_PROMPT.format(
        intent_list=INTENT_LIST,
        query=query,
    )
    try:
        res = _safe_structured_invoke(llm, prompt, IntentOutput)
        if isinstance(res, IntentOutput):
            intent = res.intent
            intent_score = res.intent_score
            should_reject = res.should_reject
            reject_reason = res.reject_reason if res.should_reject else None
        elif isinstance(res, dict):
            intent = res.get("intent", "General/Other")
            intent_score = int(res.get("intent_score", 3))
            should_reject = bool(res.get("should_reject", False))
            reject_reason = res.get("reject_reason", None) if should_reject else None
        else:
            intent = "General/Other"
            intent_score = 3
            should_reject = False
            reject_reason = None
    except Exception as e:
        logger.warning("get_intent failed: %s", e)
        intent = "General/Other"
        intent_score = 3
        should_reject = False
        reject_reason = None

    if should_reject:
        logger.info("Query REJECTED | Reason: %s | Lang: %s", reject_reason, detected_language)
        rejection_response = (
            f"I'm sorry, but I can only assist with Amazon-related queries. "
            f"Reason: {reject_reason}"
        )
        return {
            **state,
            "intent": intent,
            "intent_score": intent_score,
            "detected_language": detected_language,
            "rejected": True,
            "reject_reason": reject_reason,
            "response": rejection_response,
            "human_escalated": False,
        }

    logger.info("Intent: %s | Score: %d | Lang: %s", intent, intent_score, detected_language)
    return {
        **state,
        "intent": intent,
        "intent_score": intent_score,
        "detected_language": detected_language,
        "rejected": False,
        "reject_reason": None,
    }


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
            res = _safe_structured_invoke(llm, prompt, EscalationOutput)
            if isinstance(res, EscalationOutput):
                escalation_reason = res.escalation_reason
            elif isinstance(res, dict):
                escalation_reason = res.get("escalation_reason", "High-severity issue requiring human intervention.")
            else:
                escalation_reason = f"Escalated due to high intent score ({final_score}) for intent: {intent}"
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
    """Enhance query + generate HyDE fake answer using DeepSeek/Qwen."""
    from src.utils.llm import get_judge_llm

    llm = get_judge_llm()
    prompt = REWRITE_QUERY_PROMPT.format(
        query=state.get("query", ""),
        intent=state.get("intent", ""),
        intent_score=state.get("intent_score", 3),
    )
    try:
        res = _safe_structured_invoke(llm, prompt, RewriteQueryOutput)
        if isinstance(res, RewriteQueryOutput):
            rewritten = res.user_query_rewritten
            fake_answer = res.fake_answer
        elif isinstance(res, dict):
            rewritten = res.get("user_query_rewritten", state.get("query", ""))
            fake_answer = res.get("fake_answer", "")
        else:
            rewritten = state.get("query", "")
            fake_answer = ""
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
    Score each retrieved document 0.0–1.0.
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
        res = _safe_structured_invoke(llm, prompt, EvalRetrieverOutput)
        if isinstance(res, EvalRetrieverOutput):
            scores_list = [{"doc_index": item.doc_index, "score": item.score} for item in res.scores]
        elif isinstance(res, list):
            scores_list = res
        elif isinstance(res, dict):
            raw_scores = res.get("scores", [])
            if isinstance(raw_scores, list):
                scores_list = raw_scores
            else:
                scores_list = [res]
        else:
            scores_list = []
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
        res = _safe_structured_invoke(llm, prompt, EscalationOutput)
        if isinstance(res, EscalationOutput):
            escalation_reason = res.escalation_reason
        elif isinstance(res, dict):
            escalation_reason = res.get("escalation_reason", "No relevant information found in knowledge base.")
        else:
            escalation_reason = "No relevant information found in knowledge base."
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
        res = _safe_structured_invoke(llm, prompt, AmbiguousRewriteOutput)
        if isinstance(res, AmbiguousRewriteOutput):
            refined_query = res.refined_query
            refined_fake = res.refined_fake_answer
        elif isinstance(res, dict):
            refined_query = res.get("refined_query", state.get("user_query_rewritten", ""))
            refined_fake = res.get("refined_fake_answer", state.get("fake_answer", ""))
        else:
            refined_query = state.get("user_query_rewritten", state.get("query", ""))
            refined_fake = state.get("fake_answer", "")
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
    Use document-level scores from evaluated_docs (set by eval_retriever) to filter docs.
    Discard docs whose score is STRICTLY less than 0.3. Combine the rest into refined_context.
    Escalate to human_node only if NO doc scored >= 0.3.
    """
    from src.utils.llm import get_chat_llm

    llm = get_chat_llm()
    query = state.get("query", "")
    intent = state.get("intent", "")

    # Use already-evaluated doc scores from state (set by eval_retriever)
    evaluated_docs = state.get("evaluated_docs", [])

    if not evaluated_docs:
        # Fallback: if evaluated_docs not present, use raw documents with score 0.0
        docs = state.get("documents", [])
        evaluated_docs = [{"doc": d, "score": 0.0} for d in docs]

    # Filter: keep only docs with score >= 0.3 (discard strictly < 0.3)
    passing_docs = [e for e in evaluated_docs if e["score"] >= 0.3]
    all_scores_summary = ", ".join(f"{e['score']:.2f}" for e in evaluated_docs)

    logger.info(
        "recompose_ambiguous_strips: %d/%d docs passed score >= 0.3 (scores: [%s])",
        len(passing_docs), len(evaluated_docs), all_scores_summary
    )

    # Escalate only if NO docs have score >= 0.3
    if not passing_docs:
        strip_scores_summary = f"All evaluated docs scored below 0.3: [{all_scores_summary}]"
        prompt = AMBIGUOUS_ESCALATION_PROMPT.format(
            query=query,
            intent=intent,
            intent_score=state.get("intent_score", 3),
            strip_scores_summary=strip_scores_summary,
        )
        try:
            res = _safe_structured_invoke(llm, prompt, EscalationOutput)
            if isinstance(res, EscalationOutput):
                escalation_reason = res.escalation_reason
            elif isinstance(res, dict):
                escalation_reason = res.get("escalation_reason", "Insufficient relevant content after two-stage retrieval.")
            else:
                escalation_reason = "Insufficient relevant content after two-stage retrieval."
        except Exception as e:
            logger.warning("recompose escalation prompt failed: %s", e)
            escalation_reason = "Insufficient relevant content found after two-stage retrieval. Human agent required."

        return {**state, "escalation_reason": escalation_reason, "human_escalated": True}

    # Combine passing docs into refined context (preserve doc order by score desc)
    passing_docs_sorted = sorted(passing_docs, key=lambda e: e["score"], reverse=True)
    refined_context = "\n\n".join(e["doc"].page_content for e in passing_docs_sorted)

    logger.info(
        "recompose_ambiguous_strips: context built from %d docs, total %d chars",
        len(passing_docs), len(refined_context)
    )
    return {**state, "refined_context": refined_context, "human_escalated": False}


@traceable(name="brain_node", tags=["node", "generation"], metadata={"step": "9_brain_llm_response"})
def brain_node(state: GraphState) -> GraphState:
    """
    Final answer generation using Qwen3 with full context:
    refined_context + conversation history + intent info.
    """
    from src.utils.llm import get_chat_llm

    llm = get_chat_llm(max_tokens=1024)

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
    response = ""
    try:
        res = _safe_structured_invoke(llm, prompt, BrainNodeOutput)
        if isinstance(res, BrainNodeOutput):
            response = res.response
        elif isinstance(res, dict):
            # Normal JSON parse path: {"response": "..."}  or  {"_raw": "..."} fallback
            response = res.get("response") or res.get("_raw", "")
            # Strip any remaining JSON wrapper from raw content
            if not response and res:
                response = str(res)
        else:
            # AIMessage or plain string
            response = str(res.content if hasattr(res, "content") else res).strip()
    except Exception as e:
        logger.error("brain_node failed: %s", e)
        response = (
            "I'm sorry, I encountered an issue generating a response. "
            "Our @AmazonHelp support team is reviewing your request to assist you."
        )

    # Final guard: ensure we never return an empty or JSON-literal response
    if not response or response in ("{}", "None", "null"):
        logger.warning("brain_node: empty/invalid response from LLM, using fallback")
        response = (
            "I don't have enough information to fully resolve this directly. "
            "Please contact Amazon support at amazon.com/help or call 1-888-280-4331 "
            "for immediate assistance with your issue."
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

@traceable(name="route_after_get_intent", tags=["router", "intent"], metadata={"component": "conditional_edge"})
def route_after_get_intent(state: GraphState) -> str:
    """Route to END immediately if query was rejected, else continue to intent_evaluator."""
    if state.get("rejected", False):
        return END
    return "intent_evaluator"


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
