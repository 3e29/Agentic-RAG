"""
Result Aggregation Tools for Hadith RAG System

Implements the ResultAggregationTool for:
1. Flattening results from multiple sub-queries
2. Deduplication by hadith_id/chunk_id
3. Cross-encoder reranking for final relevance scoring

Production Standards:
- Efficient deduplication with hash maps
- Modal-deployed BGE reranker for Arabic support
- Graceful degradation without reranker
"""

import logging
import time
from typing import Dict, List, Optional, Set
from langsmith import traceable

from src.tools.retrieval.schemas import (
    Document,
    AggregatedResults,
    AggregationInput,
)

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Modal Reranker Client
# ============================================================================

_modal_reranker = None


def get_modal_reranker():
    """
    Get Modal reranker client for BGE-reranker-v2-m3.
    
    Uses Modal serverless deployment for GPU-accelerated reranking.
    No local model download needed - runs on Modal's infrastructure.
    """
    global _modal_reranker
    
    if _modal_reranker is None:
        try:
            import modal
            
            # Look up the deployed reranker using from_name (Modal 1.0+ API)
            Reranker = modal.Cls.from_name("hadith-reranker", "Reranker")
            _modal_reranker = Reranker()
            logger.info("Modal BGE reranker v2 m3 client initialized")
            
        except Exception as e:
            logger.warning(f"Modal reranker not available: {e}")
            _modal_reranker = None
    
    return _modal_reranker


# ============================================================================
# Tool Class
# ============================================================================

class ResultAggregationTool:
    """
    Aggregate and rerank results from multiple sub-queries.
    
    Handles:
    1. Flattening nested result lists
    2. Deduplication by hadith_id
    3. Score normalization
    4. Cross-encoder reranking (optional)
    """
    
    name: str = "result_aggregation"
    description: str = "Aggregate, deduplicate, and rerank search results"
    
    def __init__(self, use_reranker: bool = True):
        """
        Initialize aggregation tool.
        
        Args:
            use_reranker: Whether to use cross-encoder reranking
        """
        self.use_reranker = use_reranker
    
    @traceable(name="result_aggregation_tool")
    def __call__(
        self,
        raw_results: List[List[Document]],
        original_query: str,
        top_k: int = 20,
    ) -> AggregatedResults:
        """Aggregate and rerank results."""
        return aggregate_results(
            raw_results=raw_results,
            original_query=original_query,
            top_k=top_k,
            use_reranker=self.use_reranker,
        )


# ============================================================================
# Functional Implementation
# ============================================================================

@traceable(name="aggregate_results")
def aggregate_results(
    raw_results: List[List[Document]],
    original_query: str,
    top_k: int = 20,
    use_reranker: bool = True,
) -> AggregatedResults:
    """
    Aggregate, deduplicate, and rerank search results.
    
    Args:
        raw_results: List of result lists from multiple sub-queries
        original_query: Original user query for reranking context
        top_k: Maximum number of final results
        use_reranker: Whether to apply cross-encoder reranking
        
    Returns:
        AggregatedResults with unified ranked documents
    """
    start_time = time.time()
    logger.info(f"Aggregating results from {len(raw_results)} sub-queries")
    
    # Track sub-query contributions
    sub_query_counts: Dict[str, int] = {}
    
    # Step 1: Flatten all results
    all_documents: List[Document] = []
    for idx, result_list in enumerate(raw_results):
        sub_query_key = f"subquery_{idx}"
        sub_query_counts[sub_query_key] = len(result_list)
        all_documents.extend(result_list)
    
    total_before_dedup = len(all_documents)
    logger.info(f"Total documents before deduplication: {total_before_dedup}")
    
    # Step 2: Deduplicate by hadith_id (prefer higher scores)
    deduplicated = _deduplicate_documents(all_documents)
    duplicates_removed = total_before_dedup - len(deduplicated)
    logger.info(f"Removed {duplicates_removed} duplicates, {len(deduplicated)} unique")
    
    # Step 3: Normalize scores to [0, 1]
    normalized = _normalize_scores(deduplicated)
    
    # Step 4: Rerank with cross-encoder (optional)
    reranking_applied = False
    if use_reranker and len(normalized) > 1:
        try:
            reranked = _rerank_with_cross_encoder(normalized, original_query)
            if reranked:
                normalized = reranked
                reranking_applied = True
                logger.info("Cross-encoder reranking applied")
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
    
    # Step 5: Sort by final score and limit to top_k
    sorted_docs = sorted(normalized, key=lambda d: d.score, reverse=True)
    final_docs = sorted_docs[:top_k]
    
    execution_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"Aggregation complete: {len(final_docs)} final documents "
        f"in {execution_time:.1f}ms"
    )
    
    return AggregatedResults(
        documents=final_docs,
        total_unique=len(deduplicated),
        duplicates_removed=duplicates_removed,
        sub_query_counts=sub_query_counts,
        reranking_applied=reranking_applied,
        execution_time_ms=execution_time,
    )


def _deduplicate_documents(documents: List[Document]) -> List[Document]:
    """
    Deduplicate documents by hadith_id or chunk_id.
    
    When duplicates are found, keeps the one with highest score.
    """
    seen: Dict[str, Document] = {}
    
    for doc in documents:
        # Use parent_hadith_id if available, otherwise chunk_id
        key = doc.parent_hadith_id or doc.chunk_id
        
        if key not in seen:
            seen[key] = doc
        else:
            # Keep document with higher score
            if doc.score > seen[key].score:
                seen[key] = doc
    
    return list(seen.values())


def _normalize_scores(documents: List[Document]) -> List[Document]:
    """
    Normalize document scores to [0, 1] range.
    
    Uses min-max normalization.
    """
    if not documents:
        return []
    
    scores = [doc.score for doc in documents]
    min_score = min(scores)
    max_score = max(scores)
    
    # Handle edge case where all scores are the same
    if max_score == min_score:
        return [doc.model_copy(update={"score": 1.0}) for doc in documents]
    
    normalized = []
    for doc in documents:
        norm_score = (doc.score - min_score) / (max_score - min_score)
        normalized.append(doc.model_copy(update={"score": norm_score}))
    
    return normalized


def _rerank_with_cross_encoder(
    documents: List[Document],
    query: str,
    batch_size: int = 32,
) -> Optional[List[Document]]:
    """
    Rerank documents using Modal-deployed BGE reranker.
    
    Args:
        documents: List of documents to rerank
        query: Query for relevance scoring
        batch_size: Batch size for model inference (unused with Modal)
        
    Returns:
        Reranked documents or None if reranking fails
    """
    reranker = get_modal_reranker()
    
    if reranker is None:
        logger.warning("Modal reranker not available, skipping reranking")
        return None
    
    if not documents:
        return []
    
    try:
        # Extract passage texts
        passages = [doc.text for doc in documents]
        
        # Call Modal reranker
        scores = reranker.rerank.remote(query, passages)
        
        # Update document scores
        reranked = []
        for doc, score in zip(documents, scores):
            reranked.append(doc.model_copy(update={"score": float(score)}))
        
        logger.info(f"Modal reranker scored {len(documents)} documents")
        return reranked
        
    except Exception as e:
        logger.error(f"Modal reranking failed: {e}")
        return None


# ============================================================================
# Utility Functions
# ============================================================================

def merge_document_lists(
    list1: List[Document],
    list2: List[Document],
    prefer_higher_score: bool = True,
) -> List[Document]:
    """
    Merge two document lists with deduplication.
    
    Args:
        list1: First document list
        list2: Second document list
        prefer_higher_score: If True, keep higher-scored duplicate
        
    Returns:
        Merged and deduplicated list
    """
    combined = list1 + list2
    return _deduplicate_documents(combined)


def filter_by_score_threshold(
    documents: List[Document],
    threshold: float = 0.5,
) -> List[Document]:
    """
    Filter documents by minimum score threshold.
    
    Args:
        documents: Documents to filter
        threshold: Minimum score to keep (after normalization)
        
    Returns:
        Filtered document list
    """
    return [doc for doc in documents if doc.score >= threshold]


def boost_exact_matches(
    documents: List[Document],
    query: str,
    boost_factor: float = 1.5,
) -> List[Document]:
    """
    Boost scores for documents containing exact query matches.
    
    Args:
        documents: Documents to process
        query: Query to match
        boost_factor: Multiplier for matching documents
        
    Returns:
        Documents with boosted scores where applicable
    """
    query_lower = query.lower()
    boosted = []
    
    for doc in documents:
        if query_lower in doc.text.lower():
            new_score = min(doc.score * boost_factor, 1.0)
            boosted.append(doc.model_copy(update={"score": new_score}))
        else:
            boosted.append(doc)
    
    return boosted
