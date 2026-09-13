"""
LLM and Embeddings factory for AmazonHelp RAG Agent.

Primary: OpenRouter API for chat models (Qwen3, DeepSeek-V3)
         with langchain-openai compatible endpoint.
Fallback: HuggingFace Inference API.

Embeddings: BAAI/bge-small-en-v1.5 via HuggingFace Inference API.
"""

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── API Keys ──────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN", "")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")

# ── Model names ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# OpenRouter model IDs
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen3-8b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek/deepseek-chat-v3-0324")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@lru_cache(maxsize=4)
def get_chat_llm(model: str = None, temperature: float = 0.3, max_tokens: int = 1024):
    """Return a chat LLM for answer generation (Qwen3 via OpenRouter)."""
    model = model or CHAT_MODEL

    if OPEN_ROUTER_API_KEY:
        return _openrouter_llm(model, temperature, max_tokens)
    elif HF_TOKEN:
        return _hf_llm(model, temperature, max_tokens)
    else:
        raise ValueError(
            "No API key found. Set OPEN_ROUTER_API_KEY or HF_TOKEN in your .env file."
        )


@lru_cache(maxsize=4)
def get_judge_llm(model: str = None, temperature: float = 0.1, max_tokens: int = 512):
    """Return a judge/rewrite LLM (DeepSeek-V3 via OpenRouter)."""
    model = model or JUDGE_MODEL

    if OPEN_ROUTER_API_KEY:
        return _openrouter_llm(model, temperature, max_tokens)
    elif HF_TOKEN:
        return _hf_llm(model, temperature, max_tokens)
    else:
        raise ValueError(
            "No API key found. Set OPEN_ROUTER_API_KEY or HF_TOKEN in your .env file."
        )


def _openrouter_llm(model: str, temperature: float, max_tokens: int):
    """Create an OpenRouter LLM via ChatOpenAI-compatible interface."""
    try:
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenRouter for model: %s", model)
        return ChatOpenAI(
            model=model,
            openai_api_key=OPEN_ROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers={
                "HTTP-Referer": "https://amazonhelp-agent.local",
                "X-Title": "AmazonHelp RAG Agent",
            },
        )
    except ImportError:
        logger.warning("langchain-openai not installed, falling back to HuggingFace")
        return _hf_llm(model, temperature, max_tokens)


def _hf_llm(model: str, temperature: float, max_tokens: int):
    """Create a HuggingFace Inference API LLM."""
    from langchain_huggingface import HuggingFaceEndpoint

    # Map OpenRouter model IDs back to HF model IDs
    hf_model_map = {
        "qwen/qwen3-8b": "Qwen/Qwen3-8B",
        "deepseek/deepseek-chat-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
        "deepseek/deepseek-chat": "deepseek-ai/DeepSeek-V3",
    }
    hf_model = hf_model_map.get(model.lower(), model)

    logger.info("Using HuggingFace for model: %s", hf_model)
    return HuggingFaceEndpoint(
        repo_id=hf_model,
        task="text-generation",
        max_new_tokens=max_tokens,
        temperature=temperature,
        huggingfacehub_api_token=HF_TOKEN,
        do_sample=temperature > 0,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Return embeddings for BAAI/bge-small-en-v1.5 (local or HF API fallback)."""
    logger.info("Loading embeddings: %s", EMBEDDING_MODEL)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        logger.info("Local HuggingFaceEmbeddings unavailable (%s), trying Inference API", e)
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN is required for embeddings fallback. Set it in your .env file."
            )

        return HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=HF_TOKEN,
            model=EMBEDDING_MODEL,
        )
