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
               ├── Strip Scoring & Filtering (Qwen3)  |Re-ranking|
               └── Final Answer / Human Escalation (Qwen3)
```

**Models Used:**
| Role | Model |
|------|-------|
| Embedding | `BAAI/bge-small-en-v1.5` |
| Chat / Answer | `Qwen/Qwen3-8B` |
| Judge / Rewrite | `deepseek-ai/DeepSeek-V3-0324` |

---

## Prerequisites (Need to be installed locally to run this project)
- **Python 3.10+**
- **Docker** (for local Qdrant)

---

## Setup Instructions

```
git clone https://github.com/jamesnagar11/amazonhelp-agent.git
cd amazonhelp-agent
```

```bash
python -m venv venv
```

For Windows
```
venv\Scripts\activate
```
For Linux/Mac
```
source venv/bin/activate
```

Installing Dependencies (Might take upto ~ 6 minutes) # Please be patience
```bash
pip install -r requirements.txt
```

Configure environment variables

```bash
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac
```


### Start Qdrant (local Docker)

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v "./qdrant_storage:/qdrant/storage" --name qdrant_amazon qdrant/qdrant
```


Verify Qdrant is running:
```bash
curl http://localhost:6333/collections
# Should return {"result":{"collections":[]},"status":"ok",...}
```

### Run the ingestion pipeline (takes around ~ 4 minutes)

This embeds `data/sample_amazon_dataset.csv` into Qdrant. **Run once** — it detects if already ingested and skips.

```bash
python -m src.ingestion.ingest # make sure venv is working (active) as mentioned in previous step, or run "venv\Scripts\activate" first in windows and equivalent command in mac/linux
```

### Last Step: Launch the Streamlit app

```bash
streamlit run app.py  # make sure venv is working (active) as mentioned in previous step, or run "venv\Scripts\activate" first in windows and equivalent command in mac/linux
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
docker stop qdrant_amazon
docker rm qdrant_amazon
```

## **IMPORTANT (Please open HIVER_REPORT.md)**
To check the assignment related requirements