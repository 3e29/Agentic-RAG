"""
Embedding Helper for Modal BGE-M3 Multi-Vector Embeddings API

This module provides a robust client for the Modal-hosted BAAI/bge-m3
embedding model. It supports both dense and sparse (multi-vector) embeddings
for true hybrid search without needing a separate BM25 index.

Endpoints:
- Dense only: https://alaapocket3--bge-m3-embeddings-embed.modal.run
- Multi-vector: https://alaapocket3--bge-m3-embeddings-embed-multi-endpoint.modal.run

Model: BAAI/bge-m3
- 1024 embedding dimensions (dense)
- Sparse vectors for keyword matching (learned BM25-like)
- 8192 max tokens
- 100+ languages including Arabic
- NO prefix required (unlike E5's "passage:"/"query:")
- Best Arabic MIRACL benchmark performance

Production Standards:
- Async and sync API support
- Retry logic with exponential backoff
- LangSmith tracing via @traceable
- Type safety with proper annotations
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
import httpx
from langsmith import traceable

# Configure logging
logger = logging.getLogger(__name__)

# Modal BGE-M3 Embedding Endpoints
DENSE_ENDPOINT = "https://alaapocket3--bge-m3-embeddings-embed.modal.run"
MULTI_ENDPOINT = "https://alaapocket3--bge-m3-embeddings-embed-multi-endpoint.modal.run"

# Request configuration
DEFAULT_TIMEOUT = 120.0  # seconds (larger batches may take longer)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # exponential backoff multiplier

# Embedding dimensions
DENSE_DIMENSION = 1024


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


class ModalBGEM3Embeddings:
    """
    Client for Modal-hosted BGE-M3 multilingual embedding model.
    
    This class provides methods to generate both dense and sparse embeddings
    for text using the remote Modal API endpoints. Supports both single text
    and batch embedding generation.
    
    Key Features:
    - Dense vectors (1024-dim) for semantic search
    - Sparse vectors for keyword matching (replaces BM25)
    - Multi-vector endpoint for hybrid search in one call
    - No instruction prefix needed (unlike E5)
    
    Usage:
        embedder = ModalBGEM3Embeddings()
        
        # Dense only (fast)
        vector = embedder.embed_query("What is prayer in Islam?")
        
        # Multi-vector for hybrid search (recommended)
        dense, sparse = embedder.embed_multi_query("الصبر والشكر")
        
        # Batch
        vectors = embedder.embed_documents(["text1", "text2"])
        dense_list, sparse_list = embedder.embed_multi_documents(["text1", "text2"])
    """
    
    def __init__(
        self,
        dense_endpoint: str = DENSE_ENDPOINT,
        multi_endpoint: str = MULTI_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize the BGE-M3 embedding client.
        
        Args:
            dense_endpoint: Modal BGE-M3 dense embedding API endpoint URL
            multi_endpoint: Modal BGE-M3 multi-vector (dense+sparse) API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.dense_endpoint = dense_endpoint
        self.multi_endpoint = multi_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
    
    # =========================================================================
    # Dense Embedding Methods
    # =========================================================================
    
    @traceable(name="bge_m3_embed_query")
    def embed_query(self, text: str) -> List[float]:
        """
        Generate dense embedding for a single query text (synchronous).
        
        Args:
            text: The query text to embed
            
        Returns:
            Dense embedding vector as list of floats (1024 dimensions)
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            asyncio.get_running_loop()
            return self._embed_query_sync(text)
        except RuntimeError:
            return asyncio.run(self.aembed_query(text))
    
    def _embed_query_sync(self, text: str) -> List[float]:
        """Pure synchronous embedding using requests library."""
        import requests
        
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        logger.debug(f"Generating BGE-M3 dense embedding for: '{text[:50]}...'")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.dense_endpoint,
                    json={"texts": [text]},
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "error" in result:
                    raise EmbeddingError(f"API error: {result['error']}")
                
                if "embeddings" not in result:
                    raise EmbeddingError(f"Invalid response: missing 'embeddings' key")
                
                embeddings = result["embeddings"]
                logger.debug(f"Generated BGE-M3 embedding - dim={result.get('dimension', 1024)}")
                
                return embeddings[0] if embeddings else []
                
            except requests.HTTPError as e:
                logger.error(f"HTTP error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise EmbeddingError(f"HTTP error after {self.max_retries} retries: {e}")
                import time
                time.sleep(RETRY_BACKOFF ** attempt)
                
            except requests.RequestException as e:
                logger.error(f"Request error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise EmbeddingError(f"Request error after {self.max_retries} retries: {e}")
                import time
                time.sleep(RETRY_BACKOFF ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise EmbeddingError(f"Unexpected error after {self.max_retries} retries: {e}")
                import time
                time.sleep(RETRY_BACKOFF ** attempt)
        
        return []
    
    @traceable(name="bge_m3_embed_documents")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for multiple documents (synchronous).
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of dense embedding vectors (each 1024 dimensions)
        """
        return asyncio.run(self.aembed_documents(texts))
    
    @traceable(name="bge_m3_aembed_query")
    async def aembed_query(self, text: str) -> List[float]:
        """Generate dense embedding for a single query text (async)."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        embeddings = await self._embed_dense_texts([text])
        return embeddings[0] if embeddings else []
    
    @traceable(name="bge_m3_aembed_documents")
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for multiple documents (async)."""
        if not texts:
            return []
        
        valid_texts = [text for text in texts if text.strip()]
        return await self._embed_dense_texts(valid_texts)
    
    async def _embed_dense_texts(self, texts: List[str]) -> List[List[float]]:
        """Internal method to call the dense embedding API."""
        logger.info(f"Generating BGE-M3 dense embeddings for {len(texts)} texts")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(
                        self.dense_endpoint,
                        json={"texts": texts},
                        headers={"Content-Type": "application/json"}
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if "error" in result:
                        raise EmbeddingError(f"API error: {result['error']}")
                    
                    if "embeddings" not in result:
                        raise EmbeddingError(f"Invalid response: missing 'embeddings' key")
                    
                    embeddings = result["embeddings"]
                    logger.info(f"Generated {len(embeddings)} BGE-M3 dense embeddings (dim={result.get('dimension', 1024)})")
                    
                    return embeddings
                    
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error on attempt {attempt}/{self.max_retries}: {e.response.status_code}")
                    if attempt == self.max_retries:
                        raise EmbeddingError(f"HTTP error after {self.max_retries} retries: {e}")
                    await asyncio.sleep(RETRY_BACKOFF ** attempt)
                    
                except httpx.RequestError as e:
                    logger.error(f"Request error on attempt {attempt}/{self.max_retries}: {e}")
                    if attempt == self.max_retries:
                        raise EmbeddingError(f"Request error after {self.max_retries} retries: {e}")
                    await asyncio.sleep(RETRY_BACKOFF ** attempt)
                    
                except Exception as e:
                    logger.error(f"Unexpected error on attempt {attempt}/{self.max_retries}: {e}")
                    if attempt == self.max_retries:
                        raise EmbeddingError(f"Unexpected error after {self.max_retries} retries: {e}")
                    await asyncio.sleep(RETRY_BACKOFF ** attempt)
        
        return []
    
    # =========================================================================
    # Multi-Vector Embedding Methods (Dense + Sparse)
    # =========================================================================
    
    @traceable(name="bge_m3_embed_multi_query")
    def embed_multi_query(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """
        Generate both dense and sparse embeddings for a single query (synchronous).
        
        Args:
            text: The query text to embed
            
        Returns:
            Tuple of (dense_vector, sparse_dict)
            - dense_vector: List of floats (1024 dimensions)
            - sparse_dict: Dict mapping token_id (str) to weight (float)
        """
        try:
            asyncio.get_running_loop()
            return self._embed_multi_query_sync(text)
        except RuntimeError:
            return asyncio.run(self.aembed_multi_query(text))
    
    def _embed_multi_query_sync(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """Pure synchronous multi-vector embedding."""
        import requests
        
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        logger.debug(f"Generating BGE-M3 multi-vector embedding for: '{text[:50]}...'")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.multi_endpoint,
                    json={"texts": [text]},
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "error" in result:
                    raise EmbeddingError(f"API error: {result['error']}")
                
                dense = result.get("dense", [[]])[0]
                sparse = result.get("sparse", [{}])[0]
                
                logger.debug(f"Generated BGE-M3 multi-vector - dense_dim={len(dense)}, sparse_terms={len(sparse)}")
                
                return dense, sparse
                
            except requests.HTTPError as e:
                logger.error(f"HTTP error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise EmbeddingError(f"HTTP error after {self.max_retries} retries: {e}")
                import time
                time.sleep(RETRY_BACKOFF ** attempt)
                
            except requests.RequestException as e:
                logger.error(f"Request error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise EmbeddingError(f"Request error after {self.max_retries} retries: {e}")
                import time
                time.sleep(RETRY_BACKOFF ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise EmbeddingError(f"Unexpected error after {self.max_retries} retries: {e}")
                import time
                time.sleep(RETRY_BACKOFF ** attempt)
        
        return [], {}
    
    @traceable(name="bge_m3_embed_multi_documents")
    def embed_multi_documents(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """
        Generate both dense and sparse embeddings for multiple documents (synchronous).
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            Tuple of (dense_list, sparse_list)
            - dense_list: List of dense vectors (each 1024 floats)
            - sparse_list: List of sparse dicts (token_id -> weight)
        """
        return asyncio.run(self.aembed_multi_documents(texts))
    
    @traceable(name="bge_m3_aembed_multi_query")
    async def aembed_multi_query(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """Generate both dense and sparse embeddings for a single query (async)."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        dense_list, sparse_list = await self._embed_multi_texts([text])
        return dense_list[0] if dense_list else [], sparse_list[0] if sparse_list else {}
    
    @traceable(name="bge_m3_aembed_multi_documents")
    async def aembed_multi_documents(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """Generate both dense and sparse embeddings for multiple documents (async)."""
        if not texts:
            return [], []
        
        valid_texts = [text for text in texts if text.strip()]
        return await self._embed_multi_texts(valid_texts)
    
    async def _embed_multi_texts(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """Internal method to call the multi-vector embedding API."""
        logger.info(f"Generating BGE-M3 multi-vector embeddings for {len(texts)} texts")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(
                        self.multi_endpoint,
                        json={"texts": texts},
                        headers={"Content-Type": "application/json"}
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if "error" in result:
                        raise EmbeddingError(f"API error: {result['error']}")
                    
                    dense_list = result.get("dense", [])
                    sparse_list = result.get("sparse", [])
                    
                    logger.info(
                        f"Generated {len(dense_list)} BGE-M3 multi-vector embeddings "
                        f"(dense_dim={result.get('dimension', 1024)})"
                    )
                    
                    return dense_list, sparse_list
                    
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error on attempt {attempt}/{self.max_retries}: {e.response.status_code}")
                    if attempt == self.max_retries:
                        raise EmbeddingError(f"HTTP error after {self.max_retries} retries: {e}")
                    await asyncio.sleep(RETRY_BACKOFF ** attempt)
                    
                except httpx.RequestError as e:
                    logger.error(f"Request error on attempt {attempt}/{self.max_retries}: {e}")
                    if attempt == self.max_retries:
                        raise EmbeddingError(f"Request error after {self.max_retries} retries: {e}")
                    await asyncio.sleep(RETRY_BACKOFF ** attempt)
                    
                except Exception as e:
                    logger.error(f"Unexpected error on attempt {attempt}/{self.max_retries}: {e}")
                    if attempt == self.max_retries:
                        raise EmbeddingError(f"Unexpected error after {self.max_retries} retries: {e}")
                    await asyncio.sleep(RETRY_BACKOFF ** attempt)
        
        return [], []


# ============================================================================
# Singleton Instance (for convenience)
# ============================================================================

_default_embedder: Optional[ModalBGEM3Embeddings] = None


def get_bge_m3_embedder() -> ModalBGEM3Embeddings:
    """
    Get the default ModalBGEM3Embeddings instance (singleton).
    
    Returns:
        ModalBGEM3Embeddings instance
    """
    global _default_embedder
    
    if _default_embedder is None:
        _default_embedder = ModalBGEM3Embeddings()
        logger.info(f"Initialized Modal BGE-M3 embeddings client")
        logger.info(f"  Dense endpoint: {DENSE_ENDPOINT}")
        logger.info(f"  Multi endpoint: {MULTI_ENDPOINT}")
    
    return _default_embedder


# ============================================================================
# Convenience Functions
# ============================================================================

@traceable(name="bge_m3_embed_query")
def embed_query(text: str) -> List[float]:
    """Generate dense embedding for a query text using default BGE-M3 embedder."""
    return get_bge_m3_embedder().embed_query(text)


@traceable(name="bge_m3_embed_documents")
def embed_documents(texts: List[str]) -> List[List[float]]:
    """Generate dense embeddings for documents using default BGE-M3 embedder."""
    return get_bge_m3_embedder().embed_documents(texts)


@traceable(name="bge_m3_embed_multi_query")
def embed_multi_query(text: str) -> Tuple[List[float], Dict[str, float]]:
    """Generate multi-vector (dense + sparse) embedding for a query."""
    return get_bge_m3_embedder().embed_multi_query(text)


@traceable(name="bge_m3_embed_multi_documents")
def embed_multi_documents(texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
    """Generate multi-vector (dense + sparse) embeddings for documents."""
    return get_bge_m3_embedder().embed_multi_documents(texts)


async def aembed_query(text: str) -> List[float]:
    """Async version of embed_query."""
    return await get_bge_m3_embedder().aembed_query(text)


async def aembed_documents(texts: List[str]) -> List[List[float]]:
    """Async version of embed_documents."""
    return await get_bge_m3_embedder().aembed_documents(texts)


async def aembed_multi_query(text: str) -> Tuple[List[float], Dict[str, float]]:
    """Async version of embed_multi_query."""
    return await get_bge_m3_embedder().aembed_multi_query(text)


async def aembed_multi_documents(texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
    """Async version of embed_multi_documents."""
    return await get_bge_m3_embedder().aembed_multi_documents(texts)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    test_text = sys.argv[1] if len(sys.argv) > 1 else "أحاديث عن الصبر والشكر"
    
    print(f"\nTesting BGE-M3 embedding for: {test_text}")
    print("=" * 60)
    
    embedder = ModalBGEM3Embeddings()
    
    # Test dense embedding
    print("\n1. Testing Dense Embedding:")
    try:
        vector = embedder.embed_query(test_text)
        print(f"   ✅ Generated dense embedding with {len(vector)} dimensions")
        print(f"   First 5 values: {vector[:5]}")
        
        if len(vector) == 1024:
            print(f"   ✅ Correct dimension (1024 for BGE-M3)")
        else:
            print(f"   ⚠️  Unexpected dimension: {len(vector)}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test multi-vector embedding
    print("\n2. Testing Multi-Vector Embedding (Dense + Sparse):")
    try:
        dense, sparse = embedder.embed_multi_query(test_text)
        print(f"   ✅ Generated multi-vector embedding")
        print(f"   Dense dimension: {len(dense)}")
        print(f"   Sparse terms: {len(sparse)} non-zero")
        print(f"   Top 5 sparse terms (by weight):")
        sorted_sparse = sorted(sparse.items(), key=lambda x: x[1], reverse=True)[:5]
        for token_id, weight in sorted_sparse:
            print(f"      Token {token_id}: {weight:.4f}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test batch embedding
    print("\n3. Testing Batch Multi-Vector Embedding:")
    test_texts = [
        "الصبر نصف الإيمان",
        "Patience is half of faith",
        "حدثنا محمد بن إسماعيل",
    ]
    try:
        dense_list, sparse_list = embedder.embed_multi_documents(test_texts)
        print(f"   ✅ Generated {len(dense_list)} multi-vector embeddings")
        print(f"   Sparse terms per doc: {[len(s) for s in sparse_list]}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("BGE-M3 embedding helper test complete!")
