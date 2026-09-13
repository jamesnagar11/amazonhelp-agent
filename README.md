# AmazonHelp Customer Support AI Agent

A full-stack **RAG (Retrieval-Augmented Generation)** application that acts as an intelligent Amazon customer support agent. Built with LangGraph, Qdrant, and Streamlit.

---

## Architecture Overview

```
User Query → LangGraph CRAG Pipeline → Streamlit Frontend
               │
               ├── Intent Classification (Qwen3)
               ├── Query Rewriting + HyDE (DeepSeek-V3)
               ├── Vector Retrieval (Qdrant + BAAI/bge)
               ├── Document Evaluation (DeepSeek-V3)
               ├── Strip Scoring & Filtering (Qwen3)
               └── Final Answer / Human Escalation (Qwen3)
```

**Models Used:**
| Role | Model |
|------|-------|
| Embedding | `BAAI/bge-small-en-v1.5` |
| Chat / Answer | `Qwen/Qwen3-8B` |
| Judge / Rewrite | `deepseek-ai/DeepSeek-V3-0324` |

---

## Prerequisites

- **Python 3.10+** (tested on 3.14)
- **Docker** (for local Qdrant)
- **Hugging Face account** with a token that has Inference API access

---

## Setup Instructions

### Step 1 — Clone / Navigate to project

```bash
cd d:/langchain/project
```

### Step 2 — Create and activate virtual environment

```bash
# Create venv (skip if already exists)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables

```bash
# Copy the example file
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac
```

Open `.env` and fill in your values:

```env
# Hugging Face token (get from https://huggingface.co/settings/tokens)
HF_TOKEN=hf_your_actual_token_here

# Qdrant (leave as-is for local Docker)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=amazon_support

# Models (change only if you want different models)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHAT_MODEL=Qwen/Qwen3-8B
JUDGE_MODEL=deepseek-ai/DeepSeek-V3-0324

# Memory settings
SQLITE_PATH=./checkpoints/chat_checkpoints.db
MAX_MESSAGES=10
MAX_TOKENS=3000
```

### Step 5 — Start Qdrant (local Docker)

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant_amazon \
  qdrant/qdrant
```

> **Windows PowerShell:**
> ```powershell
> docker run -d -p 6333:6333 -p 6334:6334 `
>   -v "${PWD}/qdrant_storage:/qdrant/storage" `
>   --name qdrant_amazon `
>   qdrant/qdrant
> ```

Verify Qdrant is running:
```bash
curl http://localhost:6333/collections
# Should return {"result":{"collections":[]},"status":"ok",...}
```

### Step 6 — Run the ingestion pipeline

This embeds `data/sample_amazon_dataset.csv` into Qdrant. **Run once** — it detects if already ingested and skips.

```bash
# Activate venv first!
venv\Scripts\activate   # Windows

python -m src.ingestion.ingest
```

Expected output:
```
[INFO] Loading CSV from: data/sample_amazon_dataset.csv
[INFO] Loaded 4367 raw documents
[INFO] Split into ~18000 chunks
[INFO] Embedding dimension: 384
[INFO] Upserting in batches...
[INFO] ✅ Ingestion complete! Collection 'amazon_support' now has XXXX vectors.
```

### Step 7 — Launch the Streamlit app

```bash
venv\Scripts\streamlit run app.py   # Windows
# OR
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Project Structure

```
.
├── app.py                          # Streamlit frontend
├── requirements.txt                # Python dependencies
├── .env                            # Your API keys (never commit!)
├── .env.example                    # Template for .env
├── .gitignore                      # Excludes .env, venv, large files
├── README.md                       # This file
├── data/
│   └── sample_amazon_dataset.csv  # AmazonHelp conversation dataset
├── checkpoints/                    # Auto-created: SQLite chat memory
├── qdrant_storage/                 # Auto-created: Qdrant vector data
├── src/
│   ├── ingestion/
│   │   └── ingest.py               # CSV → Qdrant pipeline
│   ├── rag/
│   │   ├── state.py                # LangGraph state schema
│   │   ├── memory.py               # Chat memory + summarisation
│   │   ├── prompts.py              # All LLM prompt templates
│   │   ├── nodes.py                # 13 graph node functions
│   │   └── graph.py                # Graph assembly + invoke helpers
│   └── utils/
│       ├── llm.py                  # LLM + embeddings factory
│       └── vector_store.py         # Qdrant retriever factory
└── agent/
    └── agent_iteration_04.md       # This iteration's specification
```

---

## Features

| Feature | Details |
|---------|---------|
| **CRAG Pipeline** | 13-node LangGraph graph with intent classification, query rewriting, HyDE retrieval, document evaluation, strip scoring |
| **Intent Detection** | 25 AmazonHelp intents with severity scoring (1–10) |
| **Human Escalation** | Formula-based + immediate escalation for fraud/security issues |
| **Multi-Chat** | Unlimited simultaneous chats, each isolated by `thread_id` |
| **Persistent Memory** | SQLite-backed chat history that survives restarts |
| **Sliding Window** | Last 10 messages OR 3000 tokens — older messages auto-summarised by DeepSeek |
| **Streaming** | Real-time node progress shown as the graph runs |
| **Dark/Light Mode** | Toggle in the sidebar |

---

## Usage Guide

1. **Start a new chat** — Click "✨ New Chat" in the sidebar
2. **Ask a question** — Type your Amazon issue in the input box and press "Send ➤"
3. **Watch the pipeline** — Status tags show which nodes are running in real-time
4. **Resume a chat** — Click any conversation in the sidebar to continue where you left off
5. **Delete a chat** — Click 🗑 next to any conversation

---

## Stopping / Cleanup

```bash
# Stop Streamlit: Ctrl+C in terminal

# Stop Qdrant container
docker stop qdrant_amazon

# Remove Qdrant container (keeps data in qdrant_storage/)
docker rm qdrant_amazon

# To fully reset and re-ingest: delete the collection
# via Qdrant dashboard at http://localhost:6333/dashboard
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Connection refused` on Qdrant | Ensure Docker container is running: `docker ps` |
| `HF_TOKEN invalid` | Check token at huggingface.co/settings/tokens |
| `Collection is empty` | Run the ingestion script again |
| Slow responses | HF Inference API has rate limits — add `HUGGINGFACEHUB_API_TOKEN` to your `.env` |
| `ModuleNotFoundError` | Ensure venv is activated: `venv\Scripts\activate` |

---

## Iteration History

| File | Description |
|------|-------------|
| `agent/agent_iteration_01.md` | Dataset structuring (twcs.csv → filtered_amazon_dataset.csv) |
| `agent/agent_iteration_02.md` | Dataset refinement |
| `agent/agent_iteration_03.md` | Additional refinements |
| `agent/agent_iteration_04.md` | **This iteration** — full-stack RAG app |
