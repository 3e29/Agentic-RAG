"""
User Content Processing Tools for Hadith RAG System

Implements UserHadithProcessorTool (FR-RA-17):
- Process user-provided hadith text
- Index temporarily for similarity search
- Find matching authentic hadiths in knowledge base

Production Standards:
- In-memory Chroma index for temporary content
- Similarity matching with configurable threshold
- Graceful handling of non-matching content
"""

import logging
import time
from typing import List, Optional
from langsmith import traceable

from src.tools.retrieval.schemas import (
    Document,
    UserContentResult,
)
from src.tools.retrieval.search_tools import SemanticSearchTool

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Tool Class
# ============================================================================

class UserHadithProcessorTool:
    """
    Process user-provided hadith text (FR-RA-17).
    
    Handles text that users provide directly (copy-paste or typed)
    to find matching authentic hadiths in the knowledge base.
    """
    
    name: str = "user_hadith_processor"
    description: str = "Process user-provided text and find matching authentic hadiths"
    
    def __init__(
        self,
        semantic_search_tool: Optional[SemanticSearchTool] = None,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize processor.
        
        Args:
            semantic_search_tool: Tool for finding similar hadiths
            similarity_threshold: Minimum similarity for authenticity match
        """
        self._semantic_tool = semantic_search_tool or SemanticSearchTool()
        self.similarity_threshold = similarity_threshold
    
    @traceable(name="user_hadith_processor_tool")
    def __call__(
        self,
        user_text: str,
        find_similar: bool = True,
        top_k: int = 5,
    ) -> UserContentResult:
        """Process user-provided text."""
        return process_user_hadith(
            user_text=user_text,
            find_similar=find_similar,
            top_k=top_k,
            semantic_tool=self._semantic_tool,
            similarity_threshold=self.similarity_threshold,
        )


# ============================================================================
# Functional Implementation
# ============================================================================

@traceable(name="process_user_hadith")
def process_user_hadith(
    user_text: str,
    find_similar: bool = True,
    top_k: int = 5,
    semantic_tool: Optional[SemanticSearchTool] = None,
    similarity_threshold: float = 0.8,
) -> UserContentResult:
    """
    Process user-provided hadith text.
    
    This function:
    1. Cleans and normalizes the user's text
    2. Searches for similar authentic hadiths
    3. Determines if text matches an authentic hadith
    
    Args:
        user_text: Raw text provided by user
        find_similar: Whether to search for similar hadiths
        top_k: Number of similar hadiths to return
        semantic_tool: Semantic search tool instance
        similarity_threshold: Threshold for authenticity determination
        
    Returns:
        UserContentResult with processing results
    """
    start_time = time.time()
    logger.info(f"Processing user text: '{user_text[:100]}...'")
    
    # Step 1: Clean and normalize text
    processed_text = _clean_user_text(user_text)
    
    # Step 2: Find similar hadiths
    similar_hadiths: List[Document] = []
    is_authentic: Optional[bool] = None
    match_confidence: float = 0.0
    
    if find_similar and processed_text:
        if semantic_tool is None:
            semantic_tool = SemanticSearchTool()
        
        try:
            search_result = semantic_tool(
                query=processed_text,
                k=top_k,
            )
            similar_hadiths = search_result.documents
            
            # Determine authenticity match
            if similar_hadiths:
                top_score = similar_hadiths[0].score
                match_confidence = top_score
                
                if top_score >= similarity_threshold:
                    is_authentic = True
                    logger.info(f"High similarity match found: {top_score:.3f}")
                elif top_score >= 0.5:
                    is_authentic = None  # Uncertain
                    logger.info(f"Moderate similarity match: {top_score:.3f}")
                else:
                    is_authentic = False
                    logger.info(f"Low similarity - may not be authentic: {top_score:.3f}")
                    
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
    
    execution_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"User text processed in {execution_time:.1f}ms, "
        f"found {len(similar_hadiths)} similar hadiths"
    )
    
    return UserContentResult(
        original_text=user_text,
        processed_text=processed_text,
        similar_hadiths=similar_hadiths,
        is_authentic=is_authentic,
        match_confidence=match_confidence,
        indexed=False,  # We don't index user content permanently
    )


def _clean_user_text(text: str) -> str:
    """
    Clean and normalize user-provided text.
    
    Performs:
    - Whitespace normalization
    - Quote removal
    - Basic Arabic text cleaning
    """
    import re
    
    if not text:
        return ""
    
    # Remove surrounding quotes
    text = text.strip().strip('"\'""''')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common prefixes
    prefixes = [
        r'^the prophet said:?\s*',
        r'^قال رسول الله[^:]*:?\s*',
        r'^عن النبي[^:]*:?\s*',
        r'^hadith:?\s*',
        r'^حديث:?\s*',
    ]
    
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.UNICODE)
    
    return text.strip()


# ============================================================================
# Temporary Indexing (For Advanced Use Cases)
# ============================================================================

class TemporaryIndex:
    """
    In-memory index for user-provided content.
    
    Uses FAISS for efficient similarity search on temporary documents.
    """
    
    def __init__(self):
        """Initialize empty temporary index."""
        self._index = None
        self._documents: List[str] = []
        self._embeddings = None
    
    def add_document(self, text: str) -> int:
        """
        Add a document to the temporary index.
        
        Args:
            text: Document text to add
            
        Returns:
            Index of added document
        """
        self._documents.append(text)
        self._index = None  # Invalidate index
        return len(self._documents) - 1
    
    def build_index(self):
        """Build FAISS index from documents."""
        if not self._documents:
            return
        
        try:
            import faiss
            import numpy as np
            
            # Get embeddings
            model = get_embedding_model()
            texts_with_prefix = [f"passage: {t}" for t in self._documents]
            embeddings = model.encode(texts_with_prefix)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine for normalized)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self._index.add(embeddings.astype(np.float32))
            self._embeddings = embeddings
            
            logger.info(f"Built temporary index with {len(self._documents)} documents")
            
        except ImportError:
            logger.warning("FAISS not available, temporary indexing disabled")
        except Exception as e:
            logger.error(f"Failed to build temporary index: {e}")
    
    def search(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search the temporary index.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (index, score) tuples
        """
        if self._index is None:
            self.build_index()
        
        if self._index is None or not self._documents:
            return []
        
        try:
            import faiss
            import numpy as np
            
            # Embed query
            model = get_embedding_model()
            query_embedding = model.encode([f"query: {query}"])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self._index.search(
                query_embedding.astype(np.float32), 
                min(k, len(self._documents))
            )
            
            return list(zip(indices[0], scores[0]))
            
        except Exception as e:
            logger.error(f"Temporary index search failed: {e}")
            return []
    
    def clear(self):
        """Clear the temporary index."""
        self._index = None
        self._documents = []
        self._embeddings = None
