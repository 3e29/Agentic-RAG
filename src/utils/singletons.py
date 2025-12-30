"""
Singleton Pattern for Shared Resources

This module provides singleton instances for shared resources like HTTP clients
and database connections. Using singletons for these resources:
1. Reduces connection overhead (connection pooling)
2. Minimizes cold-start latency for Modal API calls
3. Ensures consistent configuration across the application

Usage:
    from src.utils.singletons import GlobalClients
    
    # Get singleton HTTP client
    client = GlobalClients.get_http_client()
    
    # Get singleton ChromaDB client
    chroma = GlobalClients.get_chroma_client()
    
    # Cleanup on shutdown
    await GlobalClients.cleanup()
"""

import asyncio
import httpx
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GlobalClients:
    """
    Singleton manager for shared client instances.
    
    Thread-safe singleton pattern using class-level locks for:
    - httpx.AsyncClient: Connection pooling for Modal API calls
    - chromadb.Client: Persistent ChromaDB connection
    
    Configuration:
    - HTTP timeout: 60s (Modal cold starts can take time)
    - HTTP connection limits: 100 max, 20 keepalive
    - ChromaDB: Persistent storage in ./data/chroma_db_bge_m3 (BGE-M3 embeddings)
    """
    
    _http_client: Optional[httpx.AsyncClient] = None
    _sync_http_client: Optional[httpx.Client] = None
    _chroma_client: Optional[chromadb.PersistentClient] = None
    _lock = asyncio.Lock()
    _sync_lock = None  # Will be threading.Lock for sync client
    _chroma_lock = None  # Will be threading.Lock for chroma client
    
    # Configuration
    HTTP_TIMEOUT = 60.0  # Modal cold starts can take up to 60s
    HTTP_MAX_CONNECTIONS = 100
    HTTP_MAX_KEEPALIVE = 20
    CHROMA_DB_PATH = "./data/chroma_db_bge_m3"  # BGE-M3 embeddings
    
    @classmethod
    async def get_http_client(cls) -> httpx.AsyncClient:
        """
        Get or create the singleton async HTTP client.
        
        Features:
        - Connection pooling for reduced latency
        - Automatic retry on transient failures (via httpx)
        - Configured timeouts for Modal API calls
        
        Returns:
            httpx.AsyncClient: Shared async HTTP client instance
        """
        if cls._http_client is None or cls._http_client.is_closed:
            async with cls._lock:
                # Double-check after acquiring lock
                if cls._http_client is None or cls._http_client.is_closed:
                    logger.info("Creating new async HTTP client (singleton)")
                    cls._http_client = httpx.AsyncClient(
                        timeout=httpx.Timeout(
                            connect=10.0,
                            read=cls.HTTP_TIMEOUT,
                            write=30.0,
                            pool=10.0
                        ),
                        limits=httpx.Limits(
                            max_connections=cls.HTTP_MAX_CONNECTIONS,
                            max_keepalive_connections=cls.HTTP_MAX_KEEPALIVE,
                            keepalive_expiry=30.0
                        ),
                        http2=True,  # HTTP/2 for multiplexing
                        follow_redirects=True
                    )
        return cls._http_client
    
    @classmethod
    def get_sync_http_client(cls) -> httpx.Client:
        """
        Get or create the singleton sync HTTP client.
        
        For use in synchronous contexts where async is not available.
        
        Returns:
            httpx.Client: Shared sync HTTP client instance
        """
        import threading
        
        if cls._sync_lock is None:
            cls._sync_lock = threading.Lock()
        
        if cls._sync_http_client is None or cls._sync_http_client.is_closed:
            with cls._sync_lock:
                # Double-check after acquiring lock
                if cls._sync_http_client is None or cls._sync_http_client.is_closed:
                    logger.info("Creating new sync HTTP client (singleton)")
                    cls._sync_http_client = httpx.Client(
                        timeout=httpx.Timeout(
                            connect=10.0,
                            read=cls.HTTP_TIMEOUT,
                            write=30.0,
                            pool=10.0
                        ),
                        limits=httpx.Limits(
                            max_connections=cls.HTTP_MAX_CONNECTIONS,
                            max_keepalive_connections=cls.HTTP_MAX_KEEPALIVE,
                            keepalive_expiry=30.0
                        ),
                        http2=True,
                        follow_redirects=True
                    )
        return cls._sync_http_client
    
    @classmethod
    def get_chroma_client(cls, persist_directory: Optional[str] = None) -> chromadb.PersistentClient:
        """
        Get or create the singleton ChromaDB client.
        
        Uses persistent storage for vector embeddings.
        
        Args:
            persist_directory: Optional custom path for ChromaDB storage.
                             Defaults to ./data/chroma_db
        
        Returns:
            chromadb.PersistentClient: Shared ChromaDB client instance
        """
        if cls._chroma_client is None:
            import threading
            
            # Use class-level lock for proper synchronization
            if cls._chroma_lock is None:
                cls._chroma_lock = threading.Lock()
            
            with cls._chroma_lock:
                if cls._chroma_client is None:
                    db_path = persist_directory or cls.CHROMA_DB_PATH
                    
                    # Ensure directory exists
                    Path(db_path).mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Creating ChromaDB client at: {db_path}")
                    cls._chroma_client = chromadb.PersistentClient(
                        path=db_path,
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
        return cls._chroma_client
    
    @classmethod
    async def cleanup(cls) -> None:
        """
        Clean up all singleton resources.
        
        Should be called during application shutdown to properly
        close connections and release resources.
        """
        logger.info("Cleaning up GlobalClients resources...")
        
        # Close async HTTP client
        if cls._http_client is not None and not cls._http_client.is_closed:
            await cls._http_client.aclose()
            cls._http_client = None
            logger.info("Async HTTP client closed")
        
        # Close sync HTTP client
        if cls._sync_http_client is not None and not cls._sync_http_client.is_closed:
            cls._sync_http_client.close()
            cls._sync_http_client = None
            logger.info("Sync HTTP client closed")
        
        # ChromaDB client doesn't need explicit cleanup
        # but we reset the reference
        cls._chroma_client = None
        logger.info("ChromaDB client reference cleared")
    
    @classmethod
    def cleanup_sync(cls) -> None:
        """
        Synchronous cleanup for use in non-async contexts.
        """
        import asyncio
        
        # Close async client if running in async context
        if cls._http_client is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(cls._http_client.aclose())
            except RuntimeError:
                # No running loop, use sync approach
                pass
            finally:
                cls._http_client = None
        
        # Close sync HTTP client
        if cls._sync_http_client is not None and not cls._sync_http_client.is_closed:
            cls._sync_http_client.close()
            cls._sync_http_client = None
        
        cls._chroma_client = None
        logger.info("GlobalClients cleaned up (sync)")
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset all singleton instances (for testing purposes).
        
        Warning: This does NOT close existing connections.
        Use cleanup() or cleanup_sync() first.
        """
        cls._http_client = None
        cls._sync_http_client = None
        cls._chroma_client = None
        logger.warning("GlobalClients reset (connections may be leaked)")


class ModalEndpoints:
    """
    Centralized Modal API endpoint configuration.
    
    Single source of truth for all Modal service URLs.
    """
    
    # LLM Endpoint (Qwen2.5-14B on A100-40GB)
    QWEN_LLM = "https://sazaitet110--qwen2-5-14b-instruct-qwenendpoint-generate.modal.run"
    
    # Embedding Endpoint (Multilingual E5 on T4)
    E5_EMBEDDINGS = "https://sazaitet110--multilingual-e5-embeddings-embed.modal.run"
    
    @classmethod
    def get_llm_url(cls) -> str:
        """Get the LLM endpoint URL."""
        return cls.QWEN_LLM
    
    @classmethod
    def get_embedding_url(cls) -> str:
        """Get the embedding endpoint URL."""
        return cls.E5_EMBEDDINGS


# Convenience functions for backwards compatibility
async def get_http_client() -> httpx.AsyncClient:
    """Convenience function to get the async HTTP client."""
    return await GlobalClients.get_http_client()


def get_sync_http_client() -> httpx.Client:
    """Convenience function to get the sync HTTP client."""
    return GlobalClients.get_sync_http_client()


def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.PersistentClient:
    """Convenience function to get the ChromaDB client."""
    return GlobalClients.get_chroma_client(persist_directory)


# Context manager for async HTTP client
class HTTPClientContext:
    """
    Async context manager that provides the singleton HTTP client.
    
    Usage:
        async with HTTPClientContext() as client:
            response = await client.post(url, json=data)
    
    Note: This does NOT close the client on exit (it's a singleton).
    """
    
    async def __aenter__(self) -> httpx.AsyncClient:
        return await GlobalClients.get_http_client()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Don't close - it's a singleton
        pass


# Export all public interfaces
__all__ = [
    "GlobalClients",
    "ModalEndpoints",
    "get_http_client",
    "get_sync_http_client", 
    "get_chroma_client",
    "HTTPClientContext"
]
