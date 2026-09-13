# Iteration 05 — Evaluation, Streaming Tokens & Feedback Loop
Created: 2026-09-13
Previous: agent_iteration_04.md

## Changes from Iteration 04

### What was kept
- Entire CRAG pipeline (all 12 nodes unchanged)
- Multi-chat memory system
- Streamlit frontend structure
- OpenRouter + HF LLM setup

### What changed / was added

## 1. RAGAS Evaluation Harness
**Why**: Iteration 04 has no way to measure quality objectively.

Use `data/golden_dataset.csv` to run automated evaluation:
- **Intent accuracy**: compare predicted intent vs. golden label
- **Answer faithfulness**: RAGAS faithfulness score (answer grounded in context)
- **Answer relevance**: RAGAS answer_relevance score
- **Escalation precision**: did we escalate the right queries?

Create `src/evaluation/evaluate.py` that:
1. Loads golden_dataset.csv
2. Runs each query through `invoke_graph()`
3. Scores with RAGAS metrics
4. Outputs a report to `evaluation_results_<timestamp>.json`

## 2. Token-Level Streaming in brain_node
**Why**: Iteration 04 streams at node-level granularity — responses feel instant only after node completion, not token by token.

Use LangChain's `.astream()` with OpenRouter's streaming support:
- `brain_node` streams tokens back via LangChain callback
- Streamlit uses `st.write_stream()` for progressive rendering
- Shows a blinking cursor while generating

## 3. User Feedback Loop
**Why**: No signal to improve retrieval quality over time.

After each AI response in the sidebar:
- 👍 / 👎 buttons
- Feedback stored in SQLite (`feedback` table)
- Low-rated responses flagged for human review
- Over time, poor-performing intents can be identified

## 4. Multi-Language Detection
**Why**: The dataset contains French, Spanish, German, Japanese queries.

Add a `detect_language` step after `get_intent`:
- Use `langdetect` or a simple HF classifier
- Pass detected language through state
- `brain_node` prompted to respond in the same language as the query

## 5. Streamlit .streamlit/config.toml
**Why**: Remove Streamlit default header/footer via config instead of CSS hacks.

```toml
[server]
headless = true

[theme]
base = "dark"
primaryColor = "#8ab4f8"
backgroundColor = "#0f0f10"
secondaryBackgroundColor = "#17181c"
textColor = "#e8eaed"
```

## Ambiguities / Issues from Iteration 04 to Fix

1. `get_retriever()` is `@lru_cache` — changing `k=4` vs `k=5` returns cached k=5 version. Fix by making the k a dynamic argument not cached.

2. The `correct` node merges correct_docs but `correct_docs` might also include docs that scored between 0.7–0.8 (ambiguous_high to correct). Should clean up the routing logic to be mutually exclusive.

3. `stream_graph` currently saves memory AFTER the full stream completes. Should save incrementally after `brain_node` update event so partial crashes don't lose data.

4. `build_context_string` puts ALL summaries + messages in every prompt. For very long chats, this will exceed context windows. Add a token budget cap per section.

## Good Luck!
