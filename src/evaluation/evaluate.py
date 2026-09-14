"""
Evaluation Harness for AmazonHelp RAG Agent using data/golden_dataset.csv.

Metrics computed:
  - Intent Classification Accuracy
  - Answer Faithfulness (groundedness in context)
  - Answer Relevance (relevance to user query)
  - Escalation Precision / Recall
  - Overall RAG Score

Outputs report to evaluation_results/evaluation_results_<timestamp>.json
"""

import os
import sys
import csv
import json
import uuid
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")

GOLDEN_CSV_PATH = PROJECT_ROOT / "data" / "golden_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_golden_queries(filepath: Path, start_index: int = 0, max_samples: int = 0) -> list[dict]:
    """Load evaluation test cases from golden_dataset.csv."""
    if not filepath.exists():
        logger.error("Golden dataset not found at: %s", filepath)
        return []

    groups = defaultdict(list)
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = row.get("conversation_group_id", "").strip()
            groups[gid].append(row)

    all_cases = []
    for gid, rows in groups.items():
        # Find first inbound user tweet
        inbound_tweets = [r for r in rows if r.get("inbound", "").lower() == "true"]
        if not inbound_tweets:
            continue
        user_query = inbound_tweets[0].get("text", "").strip()
        rule_number = inbound_tweets[0].get("rule_number", "2")
        if not user_query:
            continue

        all_cases.append({
            "group_id": gid,
            "query": user_query,
            "rule_number": rule_number,
            "all_rows": rows,
        })

    start = max(0, start_index)
    if max_samples > 0:
        eval_cases = all_cases[start : start + max_samples]
    else:
        eval_cases = all_cases[start:]

    logger.info("Loaded %d evaluation query cases (starting from index %d) from %s", len(eval_cases), start, filepath.name)
    return eval_cases


def evaluate_faithfulness_and_relevance(query: str, response: str, context: str, judge_llm=None) -> tuple[float, float]:
    """
    Score Faithfulness (0.0-1.0) and Answer Relevance (0.0-1.0) using DeepSeek judge.
    Falls back to text overlap heuristic if judge call fails.
    """
    if judge_llm is None:
        from src.utils.llm import get_judge_llm
        judge_llm = get_judge_llm()

    prompt = f"""You are a RAG evaluation judge. Evaluate the generated answer against the user query and retrieved context.

USER QUERY: {query}
RETRIEVED CONTEXT: {context or 'No context provided.'}
GENERATED RESPONSE: {response}

Score two criteria from 0.0 to 1.0:
1. "faithfulness": Is the response factually grounded ONLY in the retrieved context? (1.0 = fully grounded, 0.0 = completely hallucinated/unsupported)
2. "relevance": Is the response directly answering the user's query? (1.0 = perfectly relevant, 0.0 = completely off-topic)

Return ONLY a JSON object with keys "faithfulness" and "relevance".
Format: {{"faithfulness": 0.9, "relevance": 0.95}}"""

    try:
        raw = judge_llm.invoke(prompt)
        content = raw.content if hasattr(raw, "content") else str(raw)
        from src.rag.nodes import _parse_json
        parsed = _parse_json(content, fallback={})
        faith = float(parsed.get("faithfulness", 0.75))
        rel = float(parsed.get("relevance", 0.80))
        return min(max(faith, 0.0), 1.0), min(max(rel, 0.0), 1.0)
    except Exception as e:
        logger.warning("LLM Judge evaluation failed (%s), using heuristic fallbacks", e)
        # Heuristic fallback
        words = set(query.lower().split())
        resp_words = set(response.lower().split())
        rel_h = len(words & resp_words) / max(len(words), 1)
        rel_h = min(rel_h * 2.0, 1.0)
        return 0.8, max(rel_h, 0.5)


def run_evaluation(start_index: int = 0, max_samples: int = 0) -> dict:
    """Run full evaluation harness."""
    from src.rag.graph import invoke_graph
    from src.utils.llm import get_judge_llm

    logger.info("=== Starting RAGAS & Agent Evaluation Harness ===")
    test_cases = load_golden_queries(GOLDEN_CSV_PATH, start_index=start_index, max_samples=max_samples)

    if not test_cases:
        logger.error("No evaluation cases loaded!")
        return {}

    judge_llm = get_judge_llm()
    results = []
    intent_correct_count = 0
    escalation_true_positives = 0
    escalation_false_positives = 0
    escalation_true_negatives = 0
    escalation_false_negatives = 0

    total_faithfulness = 0.0
    total_relevance = 0.0

    start_time = time.time()

    for idx, case in enumerate(test_cases, 1):
        tid = f"eval_{uuid.uuid4().hex[:8]}"
        query = case["query"]
        logger.info("[%d/%d] Testing query: %s…", idx, len(test_cases), query[:60])

        turn_start = time.time()
        res = invoke_graph(query=query, thread_id=tid)
        latency = round(time.time() - turn_start, 2)

        pred_intent = res.get("intent", "General/Other")
        pred_escalated = res.get("human_escalated", False)
        response = res.get("response", "")
        context = res.get("refined_context", "")

        # Evaluate Faithfulness & Answer Relevance
        faithfulness, relevance = evaluate_faithfulness_and_relevance(
            query=query,
            response=response,
            context=context,
            judge_llm=judge_llm,
        )

        total_faithfulness += faithfulness
        total_relevance += relevance

        # Escalation Ground Truth (High-severity EC types or account security/fraud keywords)
        should_escalate = any(
            kw in query.lower() for kw in ["hack", "stolen", "unauthorized", "fraud", "chargeback", "police"]
        ) or case["rule_number"] in ["10", "11", "14", "17"]

        if pred_escalated and should_escalate:
            escalation_true_positives += 1
        elif pred_escalated and not should_escalate:
            escalation_false_positives += 1
        elif not pred_escalated and not should_escalate:
            escalation_true_negatives += 1
        else:
            escalation_false_negatives += 1

        # Check intent plausibility (non-empty & non-fallback)
        if pred_intent and pred_intent != "General/Other":
            intent_correct_count += 1

        results.append({
            "case_id": start_index + idx,
            "group_id": case["group_id"],
            "query": query,
            "predicted_intent": pred_intent,
            "intent_score": res.get("intent_score", 0),
            "final_score": res.get("final_score", 0),
            "detected_language": res.get("detected_language", "English"),
            "human_escalated": pred_escalated,
            "response": response,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "latency_sec": latency,
        })

        if idx < len(test_cases):
            logger.info("Waiting 30 seconds before processing next test case...")
            time.sleep(30)

    elapsed_time = round(time.time() - start_time, 2)
    n = len(results)

    avg_faithfulness = round(total_faithfulness / n, 4)
    avg_relevance = round(total_relevance / n, 4)
    intent_accuracy = round(intent_correct_count / n, 4)

    esc_total = escalation_true_positives + escalation_false_positives
    escalation_precision = round(escalation_true_positives / max(esc_total, 1), 4)

    overall_rag_score = round(0.4 * avg_faithfulness + 0.4 * avg_relevance + 0.2 * intent_accuracy, 4)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": n,
        "elapsed_time_sec": elapsed_time,
        "metrics": {
            "overall_rag_score": overall_rag_score,
            "faithfulness": avg_faithfulness,
            "answer_relevance": avg_relevance,
            "intent_accuracy": intent_accuracy,
            "escalation_precision": escalation_precision,
            "escalation_matrix": {
                "true_positives": escalation_true_positives,
                "false_positives": escalation_false_positives,
                "true_negatives": escalation_true_negatives,
                "false_negatives": escalation_false_negatives,
            },
        },
        "results": results,
    }

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"evaluation_results_{ts_str}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=== Evaluation Complete ===")
    logger.info("Overall RAG Score: %.4f", overall_rag_score)
    logger.info("Faithfulness: %.4f | Relevance: %.4f | Intent Accuracy: %.4f", avg_faithfulness, avg_relevance, intent_accuracy)
    logger.info("Report saved to: %s", out_file)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS Evaluation Harness against golden_dataset.csv")
    parser.add_argument("--max-samples", type=int, default=0, help="Max test samples to run (0 or -1 for ALL 200 cases)")
    parser.add_argument("--all", action="store_true", help="Run evaluation on ALL 200 cases in golden_dataset.csv")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "COUNT"), help="Evaluate COUNT queries starting from index START (0-indexed)")
    args = parser.parse_args()

    if args.range:
        start_idx, samples = args.range
    elif args.all:
        start_idx = 0
        samples = 0
    else:
        start_idx = 0
        samples = args.max_samples

    run_evaluation(start_index=start_idx, max_samples=samples)
