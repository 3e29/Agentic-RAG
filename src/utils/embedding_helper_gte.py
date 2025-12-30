"""
Embedding Helper for Modal GTE Embeddings API

This module provides a robust client for the Modal-hosted Alibaba-NLP/gte-multilingual-base
embedding model. It mirrors the pattern of embedding_helper.py for consistency.

Endpoint: https://sazaitet110--gte-multilingual-embeddings-embed.modal.run

Model: Alibaba-NLP/gte-multilingual-base
- 768 embedding dimensions (vs E5's 1024)
- 8192 max tokens (vs E5's 512)
- 70+ languages including Arabic
- NO prefix required (unlike E5's "passage:"/"query:")

Production Standards:
- Async and sync API support
- Retry logic with exponential backoff
- LangSmith tracing via @traceable
- Type safety with proper annotations
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import httpx
from langsmith import traceable

# Configure logging
logger = logging.getLogger(__name__)

# Modal GTE Embedding Endpoint
EMBEDDING_ENDPOINT = "https://sazaitet110--gte-multilingual-embeddings-embed.modal.run"

# Request configuration
DEFAULT_TIMEOUT = 60.0  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # exponential backoff multiplier

# Embedding dimension (gte-multilingual-base)
EMBEDDING_DIMENSION = 768


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


class ModalGTEEmbeddings:
    """
    Client for Modal-hosted GTE multilingual embedding model.
    
    This class provides methods to generate embeddings for text using
    the remote Modal API endpoint. Supports both single text and batch
    embedding generation.
    
    Key difference from E5: GTE does NOT require instruction prefixes.
    
    Usage:
        embedder = ModalGTEEmbeddings()
        
        # Single text
        vector = embedder.embed_query("What is prayer in Islam?")
        
        # Batch texts
        vectors = embedder.embed_documents(["text1", "text2"])
        
        # Async
        vector = await embedder.aembed_query("query text")
    """
    
    def __init__(
        self,
        endpoint: str = EMBEDDING_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize the GTE embedding client.
        
        Args:
            endpoint: Modal GTE embedding API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
    
    @traceable(name="gte_embed_query")
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text (synchronous).
        
        Handles both standalone sync and async-context cases.
        
        Args:
            text: The query text to embed
            
        Returns:
            Embedding vector as list of floats (768 dimensions)
            
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
        """
        import requests
        
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        logger.info(f"Generating GTE embedding (sync) for text: '{text[:50]}...'")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # GTE does NOT need any prefix - send raw text
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
                logger.info(f"Generated GTE embedding (sync) - dim={result.get('dimension', 'unknown')}")
                
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
    
    @traceable(name="gte_embed_documents")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents (synchronous).
        
        GTE does NOT require "passage:" prefix like E5.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors (each 768 dimensions)
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        return asyncio.run(self.aembed_documents(texts))
    
    @traceable(name="gte_aembed_query")
    async def aembed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text (async).
        
        GTE does NOT require any prefix - raw text is sent directly.
        
        Args:
            text: The query text to embed
            
        Returns:
            Embedding vector as list of floats (768 dimensions)
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        # GTE: NO prefix needed - send raw text
        embeddings = await self._embed_texts([text])
        return embeddings[0] if embeddings else []
    
    @traceable(name="gte_aembed_documents")
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents (async).
        
        GTE does NOT require any prefix - raw texts are sent directly.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors (each 768 dimensions)
        """
        if not texts:
            return []
        
        # GTE: NO prefix needed - send raw texts
        valid_texts = [text for text in texts if text.strip()]
        
        return await self._embed_texts(valid_texts)
    
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to call the Modal GTE embedding API with retry logic.
        
        Args:
            texts: Pre-processed texts to embed
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating GTE embeddings for {len(texts)} texts")
        
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
                        f"Generated {len(embeddings)} GTE embeddings "
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


# ============================================================================
# Singleton Instance (for convenience)
# ============================================================================

_default_embedder: Optional[ModalGTEEmbeddings] = None


def get_gte_embedder() -> ModalGTEEmbeddings:
    """
    Get the default ModalGTEEmbeddings instance (singleton).
    
    Returns:
        ModalGTEEmbeddings instance
    """
    global _default_embedder
    
    if _default_embedder is None:
        _default_embedder = ModalGTEEmbeddings()
        logger.info(f"Initialized Modal GTE embeddings client: {EMBEDDING_ENDPOINT}")
    
    return _default_embedder


# ============================================================================
# Convenience Functions
# ============================================================================

@traceable(name="gte_embed_query")
def embed_query(text: str) -> List[float]:
    """
    Generate embedding for a query text using default GTE embedder.
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector (768 dimensions)
    """
    return get_gte_embedder().embed_query(text)


@traceable(name="gte_embed_documents")
def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for documents using default GTE embedder.
    
    Args:
        texts: Document texts to embed
        
    Returns:
        List of embedding vectors (each 768 dimensions)
    """
    return get_gte_embedder().embed_documents(texts)


async def aembed_query(text: str) -> List[float]:
    """
    Async version of embed_query.
    """
    return await get_gte_embedder().aembed_query(text)


async def aembed_documents(texts: List[str]) -> List[List[float]]:
    """
    Async version of embed_documents.
    """
    return await get_gte_embedder().aembed_documents(texts)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    test_text = sys.argv[1] if len(sys.argv) > 1 else "أحاديث عن الصلاة"
    
    print(f"\nTesting GTE embedding for: {test_text}")
    print("=" * 60)
    
    embedder = ModalGTEEmbeddings()
    
    try:
        vector = embedder.embed_query(test_text)
        print(f"✅ Generated GTE embedding with {len(vector)} dimensions")
        print(f"   First 5 values: {vector[:5]}")
        
        # Verify dimension
        if len(vector) == 768:
            print(f"✅ Correct dimension (768 for GTE)")
        else:
            print(f"⚠️  Unexpected dimension: {len(vector)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
