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
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

logger = logging.getLogger(__name__)

# ── API Keys ──────────────────────────────────────────────────────────────────
HF_TOKENS_RAW = os.getenv("HF_TOKENS") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN", "")
HF_TOKENS = [t.strip() for t in HF_TOKENS_RAW.split(",") if t.strip()]
HF_TOKEN = HF_TOKENS[0] if HF_TOKENS else ""

# ── Model names ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen3.8-27B")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")


# def _hf_llm(model: str, temperature: float, max_tokens: int, token_index: int = 0):
#     """Create a HuggingFace Inference API Chat LLM using HuggingFaceEndpoint & ChatHuggingFace."""
#     from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

#     token = HF_TOKENS[token_index % len(HF_TOKENS)] if HF_TOKENS else HF_TOKEN
#     logger.info("Using HuggingFace Inference API for model: %s", model)

#     endpoint = HuggingFaceEndpoint(
#         repo_id=model,
#         task="text-generation",
#         max_new_tokens=max_tokens,
#         temperature=temperature,
#         huggingfacehub_api_token=token,
#         do_sample=temperature > 0,
#     )
#     return ChatHuggingFace(llm=endpoint)

def _hf_llm(model: str, temperature: float, max_tokens: int, token_index: int = 0):
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

    token = HF_TOKENS[token_index % len(HF_TOKENS)] if HF_TOKENS else HF_TOKEN
    logger.info("Using HuggingFace Inference API for model: %s", model)

    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        max_new_tokens=max_tokens,
        temperature=temperature,
        huggingfacehub_api_token=token,
        do_sample=temperature > 0,
    )
    return ChatHuggingFace(llm=endpoint)


def get_chat_llm(model: str = None, temperature: float = 0.3, max_tokens: int = 1024):
    """
    Return chat LLM via HuggingFace Inference API (ChatHuggingFace).
    NOTE: Not cached — different callers pass different max_tokens.
    """
    target_model = model or CHAT_MODEL
    try:
        return _hf_llm(target_model, temperature, max_tokens)
    except Exception as e:
        logger.error("get_chat_llm failed for model %s: %s", target_model, e)
        raise


def get_judge_llm(model: str = None, temperature: float = 0.1, max_tokens: int = 1024):
    """
    Return judge LLM via HuggingFace Inference API (ChatHuggingFace).
    NOTE: Not cached — different callers may need different configs.
    """
    target_model = model or JUDGE_MODEL
    try:
        return _hf_llm(target_model, temperature, max_tokens)
    except Exception as e:
        logger.error("get_judge_llm failed for model %s: %s", target_model, e)
        raise


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
