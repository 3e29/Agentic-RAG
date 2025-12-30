"""
Embedding Helper for Modal BGE-M3 Embeddings API

This module provides a robust client for the Modal-hosted BAAI/bge-m3
embedding model. Updated from multilingual-e5-large to BGE-M3 for better
Arabic/multilingual performance.

Endpoints:
- Dense only: https://alaapocket3--bge-m3-embeddings-embed.modal.run
- Multi-vector: https://alaapocket3--bge-m3-embeddings-embed-multi-endpoint.modal.run

Model: BAAI/bge-m3
- 1024 embedding dimensions (same as E5 - drop-in compatible)
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
EMBEDDING_ENDPOINT = "https://alaapocket3--bge-m3-embeddings-embed.modal.run"  # Dense only
MULTI_ENDPOINT = "https://alaapocket3--bge-m3-embeddings-embed-multi-endpoint.modal.run"  # Dense + Sparse

# Request configuration
DEFAULT_TIMEOUT = 120.0  # seconds (larger batches may take longer)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # exponential backoff multiplier

# Embedding dimension (BAAI/bge-m3 - same as E5!)
EMBEDDING_DIMENSION = 1024


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


class ModalEmbeddings:
    """
    Client for Modal-hosted BGE-M3 multilingual embedding model.
    
    This class provides methods to generate embeddings for text using
    the remote Modal API endpoint. Supports both single text and batch
    embedding generation.
    
    Updated to use BGE-M3 which:
    - Has same 1024 dimensions as E5 (drop-in replacement)
    - Best Arabic/multilingual performance (MIRACL benchmark)
    - NO prefix needed (unlike E5's "passage:"/"query:")
    - Also supports sparse vectors for hybrid search
    
    Usage:
        embedder = ModalEmbeddings()
        
        # Single text
        vector = embedder.embed_query("What is prayer in Islam?")
        
        # Batch texts
        vectors = embedder.embed_documents(["text1", "text2"])
        
        # Async
        vector = await embedder.aembed_query("query text")
        
        # Multi-vector (dense + sparse) for hybrid search
        dense, sparse = embedder.embed_multi_query("text")
    """
    
    def __init__(
        self,
        endpoint: str = EMBEDDING_ENDPOINT,
        multi_endpoint: str = MULTI_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize the embedding client.
        
        Args:
            endpoint: Modal BGE-M3 dense embedding API endpoint URL
            multi_endpoint: Modal BGE-M3 multi-vector (dense+sparse) API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.endpoint = endpoint
        self.multi_endpoint = multi_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
    
    @traceable(name="embed_query")
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text (synchronous).
        
        BGE-M3 does NOT need query/passage prefixes unlike E5.
        Handles both standalone sync and async-context cases.
        
        Args:
            text: The query text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # In async context - use sync HTTP call
            return self._embed_query_sync(text)
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            return asyncio.run(self.aembed_query(text))
    
    def _embed_query_sync(self, text: str) -> List[float]:
        """
        Pure synchronous embedding using requests library.
        
        Used when called from within an async context where asyncio.run() would fail.
        BGE-M3 does NOT need any prefixes.
        """
        import requests
        
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        logger.info(f"Generating BGE-M3 embedding (sync) for text: '{text[:50]}...'")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
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
                logger.info(f"Generated BGE-M3 embedding (sync) - dim={result.get('dimension', 'unknown')}")
                
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
    
    @traceable(name="embed_documents")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents (synchronous).
        
        BGE-M3 does NOT need passage prefixes unlike E5.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        return asyncio.run(self.aembed_documents(texts))
    
    @traceable(name="aembed_query")
    async def aembed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text (async).
        
        BGE-M3 does NOT need query/passage prefixes unlike E5.
        
        Args:
            text: The query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        # BGE-M3: No prefix needed
        embeddings = await self._embed_texts([text])
        return embeddings[0] if embeddings else []
    
    @traceable(name="aembed_documents")
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents (async).
        
        BGE-M3 does NOT need passage prefixes unlike E5.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # BGE-M3: No prefix needed
        valid_texts = [text for text in texts if text.strip()]
        
        return await self._embed_texts(valid_texts)
    
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to call the Modal BGE-M3 embedding API with retry logic.
        
        Args:
            texts: Pre-processed texts to embed
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating BGE-M3 embeddings for {len(texts)} texts")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(
                        self.endpoint,
                        json={"texts": texts},
                        headers={"Content-Type": "application/json"}
                    )
                    
                    # Check for HTTP errors
                    response.raise_for_status()
                    
                    # Parse response
                    result = response.json()
                    
                    # Handle error response from API
                    if "error" in result:
                        raise EmbeddingError(f"API error: {result['error']}")
                    
                    # Validate response structure
                    if "embeddings" not in result:
                        raise EmbeddingError(f"Invalid response: missing 'embeddings' key")
                    
                    embeddings = result["embeddings"]
                    
                    logger.info(
                        f"Generated {len(embeddings)} embeddings "
                        f"(dim={result.get('dimension', 'unknown')})"
                    )
                    
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
        
        # Should never reach here, but return empty list as fallback
        return []

    # ========================================================================
    # Multi-Vector Methods (Dense + Sparse for Hybrid Search)
    # ========================================================================
    
    @traceable(name="embed_multi_query")
    def embed_multi_query(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """
        Generate both dense and sparse embeddings for a single query (sync).
        
        The sparse embedding can be used for keyword matching, replacing BM25
        with BGE-M3's learned sparse representation.
        
        Args:
            text: The query text to embed
            
        Returns:
            Tuple of (dense_vector, sparse_dict) where:
                - dense_vector: 1024-dim embedding
                - sparse_dict: {token_id: weight} for keyword matching
        """
        try:
            asyncio.get_running_loop()
            return self._embed_multi_query_sync(text)
        except RuntimeError:
            return asyncio.run(self.aembed_multi_query(text))
    
    def _embed_multi_query_sync(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """Pure sync version using requests library."""
        import requests
        
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        logger.info(f"Generating BGE-M3 multi-vector (sync) for: '{text[:50]}...'")
        
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
                
                dense = result["dense_embeddings"][0]
                sparse = result["sparse_embeddings"][0]
                
                logger.info(f"Generated multi-vector: dense={len(dense)}d, sparse={len(sparse)} terms")
                return dense, sparse
                
            except Exception as e:
                if attempt == self.max_retries:
                    raise EmbeddingError(f"Error after {self.max_retries} retries: {e}")
                import time
                time.sleep(RETRY_BACKOFF ** attempt)
        
        return [], {}
    
    @traceable(name="aembed_multi_query")
    async def aembed_multi_query(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """
        Generate both dense and sparse embeddings for a single query (async).
        
        Args:
            text: The query text to embed
            
        Returns:
            Tuple of (dense_vector, sparse_dict)
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        result = await self._embed_multi_texts([text])
        if result[0] and result[1]:
            return result[0][0], result[1][0]
        return [], {}
    
    @traceable(name="embed_multi_documents")
    def embed_multi_documents(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """
        Generate both dense and sparse embeddings for multiple documents (sync).
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            Tuple of (dense_vectors, sparse_dicts)
        """
        return asyncio.run(self.aembed_multi_documents(texts))
    
    @traceable(name="aembed_multi_documents")
    async def aembed_multi_documents(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """
        Generate both dense and sparse embeddings for multiple documents (async).
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            Tuple of (dense_vectors, sparse_dicts)
        """
        if not texts:
            return [], []
        
        valid_texts = [text for text in texts if text.strip()]
        return await self._embed_multi_texts(valid_texts)
    
    async def _embed_multi_texts(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """
        Internal method to call the BGE-M3 multi-vector endpoint.
        
        Args:
            texts: Pre-processed texts to embed
            
        Returns:
            Tuple of (dense_vectors, sparse_dicts)
        """
        logger.info(f"Generating BGE-M3 multi-vectors for {len(texts)} texts")
        
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
                    
                    dense = result["dense_embeddings"]
                    sparse = result["sparse_embeddings"]
                    
                    logger.info(
                        f"Generated {len(dense)} multi-vectors "
                        f"(dense={result.get('dense_dimension', 'unknown')}d)"
                    )
                    
                    return dense, sparse
                    
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error on attempt {attempt}: {e.response.status_code}")
                    if attempt == self.max_retries:
                        raise EmbeddingError(f"HTTP error after {self.max_retries} retries: {e}")
                    await asyncio.sleep(RETRY_BACKOFF ** attempt)
                    
                except Exception as e:
                    logger.error(f"Error on attempt {attempt}: {e}")
                    if attempt == self.max_retries:
                        raise EmbeddingError(f"Error after {self.max_retries} retries: {e}")
                    await asyncio.sleep(RETRY_BACKOFF ** attempt)
        
        return [], []


# ============================================================================
# Singleton Instance (for convenience)
# ============================================================================

_default_embedder: Optional[ModalEmbeddings] = None


def get_embedder() -> ModalEmbeddings:
    """
    Get the default ModalEmbeddings instance (singleton).
    
    Returns:
        ModalEmbeddings instance
    """
    global _default_embedder
    
    if _default_embedder is None:
        _default_embedder = ModalEmbeddings()
        logger.info(f"Initialized Modal embeddings client: {EMBEDDING_ENDPOINT}")
    
    return _default_embedder


# ============================================================================
# Convenience Functions
# ============================================================================

@traceable(name="embed_query")
def embed_query(text: str) -> List[float]:
    """
    Generate embedding for a query text using default embedder.
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector
    """
    return get_embedder().embed_query(text)


@traceable(name="embed_documents")
def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for documents using default embedder.
    
    Args:
        texts: Document texts to embed
        
    Returns:
        List of embedding vectors
    """
    return get_embedder().embed_documents(texts)


async def aembed_query(text: str) -> List[float]:
    """
    Async version of embed_query.
    """
    return await get_embedder().aembed_query(text)


async def aembed_documents(texts: List[str]) -> List[List[float]]:
    """
    Async version of embed_documents.
    """
    return await get_embedder().aembed_documents(texts)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    test_text = sys.argv[1] if len(sys.argv) > 1 else "أحاديث عن الصلاة"
    
    print(f"\nTesting embedding for: {test_text}")
    print("=" * 60)
    
    embedder = ModalEmbeddings()
    
    try:
        vector = embedder.embed_query(test_text)
        print(f"✅ Generated embedding with {len(vector)} dimensions")
        print(f"   First 5 values: {vector[:5]}")
    except Exception as e:
        print(f"❌ Error: {e}")
