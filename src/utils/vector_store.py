"""Qdrant vector store factory with index caching."""

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "amazon_support")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_URL = os.getenv("QDRANT_URL", None)


@lru_cache(maxsize=1)
def get_qdrant_client():
    """Return a cached Qdrant client (supports local disk, local Docker, or Qdrant Cloud)."""
    from qdrant_client import QdrantClient

    # 1. Qdrant Cloud URL + API Key
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # 2. Local Disk fallback or explicit local setting
    if os.getenv("USE_LOCAL_QDRANT", "true").lower() == "true":
        return QdrantClient(path=QDRANT_PATH)

    # 3. Remote Host + Port fallback
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=3.0)
        client.get_collections()
        return client
    except Exception:
        return QdrantClient(path=QDRANT_PATH)


def get_vector_store():
    """Return a LangChain Qdrant vector store instance."""
    from langchain_qdrant import QdrantVectorStore
    from src.utils.llm import get_embeddings

    client = get_qdrant_client()
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )


from langsmith import traceable


@lru_cache(maxsize=16)
@traceable(name="get_retriever", tags=["vector_store", "qdrant"], metadata={"component": "retriever_factory"})
def get_retriever(k: int = 5):
    """Return a cached Qdrant retriever for top-k similarity search."""
    vs = get_vector_store()
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})


def collection_exists() -> bool:
    """Check if the Qdrant collection already has vectors (index cached)."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION)
        return getattr(info, "vectors_count", 0) > 0 or getattr(info, "points_count", 0) > 0
    except Exception:
        return False
