"""
Search Orchestrator - Parallel Search Execution Service

This module orchestrates parallel search execution across sub-queries and
handles result aggregation. It separates orchestration concerns from the
LLM decision-making logic in RetrievalAgent.

Responsibilities:
- Parallel execution of ReAct agents for sub-queries
- Result aggregation and merging from multiple searches
- Cross-query deduplication
- Coordination of the search workflow

Production Standards:
- Single Responsibility: Only orchestration, no LLM logic
- Async-first: Uses asyncio for parallel execution
- Dependency Injection: Repository and agent executor can be injected
- Observable: Full tracing via metadata collection

Usage:
    from src.agents.search_orchestrator import SearchOrchestrator
    
    orchestrator = SearchOrchestrator(repository=hadith_repo)
    results = await orchestrator.execute_parallel_searches(
        sub_queries=["query1", "query2"],
        target_collections=["bukhari", "muslim"],
    )
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Awaitable

from src.tools.retrieval.schemas import Document, AggregatedResults
from src.tools.retrieval.aggregation_tools import aggregate_results
from src.data.hadith_repository import HadithRepository, get_hadith_repository
from src.config.settings import DEFAULT_TOP_K, PARALLEL_SEARCH_K

logger = logging.getLogger(__name__)


class SearchOrchestrator:
    """
    Orchestrates parallel search execution and result aggregation.
    
    This service handles:
    - Parallel execution of autonomous agents for sub-queries
    - Result aggregation with cross-encoder reranking
    - Cross-query deduplication
    - Chunk reassembly coordination
    
    Attributes:
        _repository: HadithRepository for data access
        _agent_executor: Callable for executing individual agent searches
    """
    
    def __init__(
        self,
        repository: Optional[HadithRepository] = None,
        agent_executor: Optional[Callable[..., Awaitable[Tuple[List[Document], List[Dict]]]]] = None,
    ):
        """
        Initialize the orchestrator with optional dependency injection.
        
        Args:
            repository: HadithRepository instance. Uses singleton if not provided.
            agent_executor: Async callable for executing individual agent searches.
                           If not provided, must be set before calling execute methods.
        """
        self._repository = repository
        self._agent_executor = agent_executor
    
    @property
    def repository(self) -> HadithRepository:
        """Get the repository, initializing from singleton if needed."""
        if self._repository is None:
            self._repository = get_hadith_repository()
        return self._repository
    
    def set_agent_executor(
        self,
        executor: Callable[..., Awaitable[Tuple[List[Document], List[Dict]]]],
    ) -> None:
        """
        Set the agent executor function.
        
        Args:
            executor: Async callable that executes a single agent search.
                     Signature: (query, collections, language, output_lang, intent, index) -> (docs, logs)
        """
        self._agent_executor = executor
    
    async def execute_parallel_searches(
        self,
        sub_queries: List[str],
        target_collections: List[str],
        query_language: str = "ar",
        desired_output_language: Optional[str] = None,
        query_intent: str = "thematic_search",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[List[Document]], List[Dict[str, Any]]]:
        """
        Execute autonomous agents for all sub-queries in parallel.
        
        Each sub-query gets its own ReAct agent that runs independently.
        Results are collected and returned along with iteration logs.
        
        Args:
            sub_queries: List of queries to search for
            target_collections: Collections to search (bukhari, muslim)
            query_language: Detected language of the query (ar/en/mixed)
            desired_output_language: User's explicit preference for result language
            query_intent: Intent of the query (specific_lookup, thematic_search, etc.)
            metadata: Optional dict to store iteration logs
            
        Returns:
            Tuple of (list of document lists per query, all iteration logs)
        """
        if not self._agent_executor:
            raise RuntimeError(
                "Agent executor not set. Call set_agent_executor() first "
                "or provide executor in constructor."
            )
        
        logger.info(f"Executing {len(sub_queries)} sub-queries in parallel")
        
        # Create async tasks for each sub-query
        tasks = [
            self._agent_executor(
                query=sub_query,
                target_collections=target_collections,
                query_language=query_language,
                desired_output_language=desired_output_language,
                query_intent=query_intent,
                query_index=i,
            )
            for i, sub_query in enumerate(sub_queries)
        ]
        
        # Execute all agents in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results: List[List[Document]] = []
        all_iteration_logs: List[Dict[str, Any]] = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent for sub-query {i} failed: {result}")
                processed_results.append([])
            else:
                docs, iteration_logs = result
                processed_results.append(docs)
                all_iteration_logs.extend(iteration_logs)
        
        # Store iteration logs in metadata if provided
        if metadata is not None:
            metadata["agent_iterations"] = all_iteration_logs
        
        return processed_results, all_iteration_logs
    
    def aggregate_sub_query_results(
        self,
        sub_queries: List[str],
        all_results: List[List[Document]],
        per_query_top_k: int = DEFAULT_TOP_K,
        use_reranker: bool = True,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Aggregate results from multiple sub-queries.
        
        Each sub-query's results are reranked against that sub-query,
        then merged and deduplicated across all sub-queries.
        
        Args:
            sub_queries: The original sub-queries
            all_results: List of document lists (one per sub-query)
            per_query_top_k: Number of results to keep per sub-query
            use_reranker: Whether to apply cross-encoder reranking
            
        Returns:
            Tuple of (merged documents, aggregation statistics)
        """
        logger.info(
            f"Aggregating results: {len(sub_queries)} sub-queries × {per_query_top_k} results each"
        )
        
        aggregation_stats = {
            "per_query_results": [],
            "total_unique": 0,
            "duplicates_removed": 0,
            "reranking_applied": False,
        }
        
        merged_documents: List[Document] = []
        
        for i, (sub_query, docs) in enumerate(zip(sub_queries, all_results)):
            if not docs:
                logger.warning(f"Sub-query {i} returned no results: '{sub_query[:50]}...'")
                aggregation_stats["per_query_results"].append({
                    "sub_query": sub_query[:100],
                    "raw_count": 0,
                    "after_rerank": 0,
                })
                continue
            
            # Aggregate this sub-query's results against its own query
            sub_aggregated = aggregate_results(
                raw_results=[docs],
                original_query=sub_query,
                top_k=per_query_top_k,
                use_reranker=use_reranker,
            )
            
            aggregation_stats["per_query_results"].append({
                "sub_query": sub_query[:100],
                "raw_count": len(docs),
                "after_rerank": len(sub_aggregated.documents),
            })
            aggregation_stats["total_unique"] += sub_aggregated.total_unique
            aggregation_stats["duplicates_removed"] += sub_aggregated.duplicates_removed
            aggregation_stats["reranking_applied"] = (
                aggregation_stats["reranking_applied"] or sub_aggregated.reranking_applied
            )
            
            merged_documents.extend(sub_aggregated.documents)
            logger.info(f"Sub-query {i}: {len(docs)} raw → {len(sub_aggregated.documents)} after rerank")
        
        # Final deduplication across all sub-queries
        final_documents, cross_duplicates = self._deduplicate_documents(merged_documents)
        aggregation_stats["cross_query_duplicates_removed"] = cross_duplicates
        
        logger.info(
            f"Final merge: {len(merged_documents)} → {len(final_documents)} "
            f"(removed {cross_duplicates} cross-query duplicates)"
        )
        
        return final_documents, aggregation_stats
    
    def _deduplicate_documents(
        self,
        documents: List[Document],
    ) -> Tuple[List[Document], int]:
        """
        Deduplicate documents across sub-queries.
        
        Uses parent_hadith_id or chunk_id as the unique key.
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            Tuple of (deduplicated documents, count of duplicates removed)
        """
        seen_ids = set()
        unique_docs: List[Document] = []
        
        for doc in documents:
            doc_key = doc.parent_hadith_id or doc.chunk_id
            if doc_key not in seen_ids:
                seen_ids.add(doc_key)
                unique_docs.append(doc)
        
        duplicates_removed = len(documents) - len(unique_docs)
        return unique_docs, duplicates_removed
    
    def reassemble_and_finalize(
        self,
        documents: List[Document],
        desired_output_language: Optional[str] = None,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Reassemble chunked hadiths and finalize the result set.
        
        Args:
            documents: Documents to process
            desired_output_language: Language preference for chunks
            
        Returns:
            Tuple of (reassembled documents, reassembly statistics)
        """
        input_count = len(documents)
        
        reassembled_docs = self.repository.reassemble_chunked_hadiths(
            documents=documents,
            desired_language=desired_output_language,
        )
        
        reassembly_stats = {
            "input_count": input_count,
            "output_count": len(reassembled_docs),
        }
        
        return reassembled_docs, reassembly_stats
    
    async def orchestrate_search(
        self,
        sub_queries: List[str],
        target_collections: List[str],
        query_language: str = "ar",
        desired_output_language: Optional[str] = None,
        query_intent: str = "thematic_search",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Full orchestration pipeline: parallel search → aggregate → reassemble.
        
        This is the main entry point for the orchestrator, combining all steps:
        1. Execute parallel searches for all sub-queries
        2. Aggregate and rerank results per sub-query
        3. Merge and deduplicate across sub-queries
        4. Reassemble chunked hadiths
        
        Args:
            sub_queries: Queries to search for
            target_collections: Collections to search
            query_language: Detected query language
            desired_output_language: User's language preference
            query_intent: Query intent from analysis
            metadata: Dict to store execution metadata
            
        Returns:
            Final list of reassembled, deduplicated documents
        """
        if metadata is None:
            metadata = {}
        
        # Step 1: Execute parallel searches
        all_results, iteration_logs = await self.execute_parallel_searches(
            sub_queries=sub_queries,
            target_collections=target_collections,
            query_language=query_language,
            desired_output_language=desired_output_language,
            query_intent=query_intent,
            metadata=metadata,
        )
        
        # Step 2: Aggregate results
        merged_docs, aggregation_stats = self.aggregate_sub_query_results(
            sub_queries=sub_queries,
            all_results=all_results,
            per_query_top_k=DEFAULT_TOP_K,
            use_reranker=True,
        )
        metadata["aggregation"] = aggregation_stats
        
        # Step 3: Reassemble chunks
        final_docs, reassembly_stats = self.reassemble_and_finalize(
            documents=merged_docs,
            desired_output_language=desired_output_language,
        )
        metadata["reassembly"] = reassembly_stats
        
        return final_docs


# ============================================================================
# Module-level convenience functions (backward compatibility)
# ============================================================================

# Singleton orchestrator instance
_default_orchestrator: Optional[SearchOrchestrator] = None


def get_search_orchestrator() -> SearchOrchestrator:
    """
    Get the default singleton SearchOrchestrator instance.
    
    Note: The agent executor must still be set before use.
    
    Returns:
        SearchOrchestrator: Singleton instance
    """
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = SearchOrchestrator()
    return _default_orchestrator
