"""
Retrieval Agent for Hadith RAG System

This module implements the Retrieval Agent with two modes:
1. **Autonomous Mode (ReAct Pattern)** - LLM dynamically decides which tools to use
2. **Chain Mode** - Fixed Map-Reduce pipeline (legacy)

**Autonomous Agent Architecture:**
- Uses ReAct (Reasoning + Acting) pattern
- LLM decides: expand query? extract filters? which search tool? retry?
- Iterative loop until sufficient results or max attempts

**Key Features:**
- Dynamic tool selection based on query characteristics
- Self-correction: Agent decides when to relax filters
- Observable in LangSmith as "agent" type (not "chain")
- Graceful fallbacks at every stage

**Production Standards:**
- LangGraph integration for workflow orchestration
- Pydantic V2 for data validation
- Dependency injection for vector store/BM25
- Full observability via logging and LangSmith tracing
- Zero hallucination: Only returns raw documents
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from langsmith import traceable

from src.graph.state import AgentState, InputSource
from src.tools.retrieval.schemas import (
    Document,
    MetadataFilter,
    AggregatedResults,
    SearchResult,
    HybridSearchResult,
)
from src.tools.retrieval.search_tools import (
    HybridSearchTool,
    SemanticSearchTool,
    KeywordSearchTool,
    hybrid_search,
    semantic_search,
    keyword_search,
)
from src.tools.retrieval.filter_tools import (
    MetadataFilterTool,
    QueryExpansionTool,
    extract_metadata_filters,
    expand_query,
)
from src.tools.retrieval.chapter_tools import (
    ChapterLookupTool,
    find_chapter_for_subject,
)
from src.utils.prompts import format_prompt
from src.utils.llm_helper import call_llm, call_llm_sync, parse_json_response
from src.tools.retrieval.aggregation_tools import (
    ResultAggregationTool,
    aggregate_results,
)
from src.tools.retrieval.user_content_tools import (
    UserHadithProcessorTool,
    process_user_hadith,
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MAX_AGENT_ITERATIONS = 5  # Max ReAct loop iterations
MAX_RETRIES = 3
DEFAULT_TOP_K = 5  # Final results after reranking
PARALLEL_SEARCH_K = 50  # Fetch 50 hadiths per search for better cross-encoder reranking


# ============================================================================
# Chunk Reassembly Helper
# ============================================================================

def _reassemble_chunked_hadiths(
    documents: List[Document],
    desired_language: Optional[str] = None,
) -> List[Document]:
    """
    Reassemble chunked hadiths into their complete form.
    
    When a hadith is too long and was split into multiple chunks during embedding,
    this function fetches all chunks and combines them into the full hadith text.
    
    Args:
        documents: List of Document objects from search results
        desired_language: Language preference for fetching chunks (arabic/english)
        
    Returns:
        List of Document objects with chunked hadiths reassembled
    """
    from src.utils.singletons import get_chroma_client
    
    if not documents:
        return documents
    
    # Identify documents that need reassembly (total_chunks > 1 and we only have 1 chunk)
    reassembly_needed = []
    for doc in documents:
        if doc.total_chunks and doc.total_chunks > 1:
            reassembly_needed.append(doc)
    
    if not reassembly_needed:
        return documents
    
    logger.info(f"Reassembling {len(reassembly_needed)} chunked hadiths")
    
    try:
        client = get_chroma_client()
        reassembled_docs = []
        reassembled_hadith_ids = set()  # Track which hadiths we've already reassembled
        
        for doc in documents:
            # Skip if already reassembled this hadith
            hadith_key = (doc.hadith_id, doc.language, doc.collection)
            if hadith_key in reassembled_hadith_ids:
                continue
            
            # If not chunked, keep as-is
            if not doc.total_chunks or doc.total_chunks <= 1:
                reassembled_docs.append(doc)
                reassembled_hadith_ids.add(hadith_key)
                continue
            
            # Need to fetch all chunks for this hadith
            # Normalize collection name to database format
            coll_name = doc.collection.lower() if doc.collection else "bukhari"
            if "bukhari" in coll_name:
                coll_name = "bukhari"
            elif "muslim" in coll_name:
                coll_name = "muslim"
            collection_db_name = f"hadith_{coll_name}"
            
            try:
                collection = client.get_collection(collection_db_name)
                
                # Build query filter
                lang_filter = doc.language or desired_language or "arabic"
                all_chunks = collection.get(
                    where={
                        "$and": [
                            {"hadith_id": {"$eq": doc.hadith_id}},
                            {"language": {"$eq": lang_filter}}
                        ]
                    },
                    include=['metadatas', 'documents']
                )
                
                if not all_chunks['ids']:
                    # Fallback: keep original chunk
                    reassembled_docs.append(doc)
                    reassembled_hadith_ids.add(hadith_key)
                    continue
                
                # Sort chunks by chunk_index
                chunk_data = list(zip(
                    all_chunks['ids'],
                    all_chunks['metadatas'],
                    all_chunks['documents']
                ))
                chunk_data.sort(key=lambda x: x[1].get('chunk_index', 0))
                
                # Combine all chunk texts
                combined_text = "\n".join([text for _, _, text in chunk_data])
                
                # Create new Document with combined text
                reassembled_doc = Document(
                    chunk_id=doc.chunk_id,  # Keep original chunk_id for reference
                    text=combined_text,
                    score=doc.score,
                    search_type=doc.search_type,
                    language=doc.language,
                    collection=doc.collection,
                    book_id=doc.book_id,
                    chapter_id=doc.chapter_id,
                    hadith_id=doc.hadith_id,
                    narrator=doc.narrator,
                    parent_hadith_id=doc.parent_hadith_id,
                    book_number=doc.book_number,
                    chapter_number=doc.chapter_number,
                    hadith_id_in_book=doc.hadith_id_in_book,
                    chunk_index=0,  # Now it's a complete document
                    total_chunks=doc.total_chunks,
                    is_chunked=False,  # No longer chunked - it's complete
                )
                reassembled_docs.append(reassembled_doc)
                reassembled_hadith_ids.add(hadith_key)
                
                logger.debug(
                    f"Reassembled Hadith #{doc.hadith_id}: {len(chunk_data)} chunks -> {len(combined_text)} chars"
                )
                
            except Exception as e:
                logger.warning(f"Failed to reassemble hadith {doc.hadith_id}: {e}")
                # Keep original chunk on error
                if hadith_key not in reassembled_hadith_ids:
                    reassembled_docs.append(doc)
                    reassembled_hadith_ids.add(hadith_key)
        
        logger.info(f"Reassembly complete: {len(documents)} -> {len(reassembled_docs)} documents")
        return reassembled_docs
        
    except Exception as e:
        logger.error(f"Chunk reassembly failed: {e}")
        return documents  # Return original on failure


# ============================================================================
# Autonomous Retrieval Agent (ReAct Pattern)
# ============================================================================

@traceable(name="retrieval_agent", run_type="chain")
def retrieval_agent(state: AgentState) -> Dict[str, Any]:
    """
    Main Retrieval Agent node for LangGraph workflow.
    
    Uses ReAct (Reasoning + Acting) pattern where the LLM autonomously decides:
    - Whether to expand the query
    - Whether to extract metadata filters  
    - Which search tool to use (keyword/semantic/hybrid)
    - Whether to retry with relaxed filters
    - When to stop searching
    
    Args:
        state: Current AgentState with query analysis results
        
    Returns:
        Dictionary with 'retrieved_docs' and updated 'metadata'
    """
    start_time = time.time()
    
    # Get query from state
    query = (
        state.get("corrected_query") or 
        state.get("normalized_query") or 
        state.get("original_query", "")
    )
    
    if not query:
        logger.error("Retrieval agent called with no query")
        return {
            "retrieved_docs": [],
            "metadata": _update_metadata(state, {"error": "No query provided"})
        }
    
    logger.info(f"Starting autonomous retrieval for: '{query[:100]}...'")
    
    # Initialize metadata
    retrieval_metadata = {
        "agent": "retrieval_autonomous",
        "query_used": query,
        "agent_iterations": [],
        "errors": [],
        "stages": [],  # Track processing stages
    }
    
    # Route based on input source
    input_source = state.get("input_source", "base_knowledge")
    query_intent = state.get("query_intent", "thematic_search")
    
    if input_source == "user_text":
        result = _handle_user_text(query, state, retrieval_metadata)
    elif input_source == "file_upload":
        result = _handle_user_text(query, state, retrieval_metadata)
    elif query_intent == "metadata_query":
        # Handle metadata-based queries (longest, shortest, most, count, etc.)
        result = _handle_metadata_query(query, state, retrieval_metadata)
    else:
        # Use autonomous agent for base knowledge search
        result = _autonomous_search(query, state, retrieval_metadata)
    
    execution_time = (time.time() - start_time) * 1000
    retrieval_metadata["total_execution_time_ms"] = execution_time
    
    logger.info(
        f"Autonomous retrieval complete: {len(result)} documents in {execution_time:.1f}ms"
    )
    
    return {
        "retrieved_docs": result,
        "metadata": _update_metadata(state, {"retrieval": retrieval_metadata})
    }


@traceable(name="autonomous_search_loop", run_type="chain")
def _autonomous_search(
    query: str,
    state: AgentState,
    metadata: Dict[str, Any],
) -> List[Document]:
    """
    Autonomous search using ReAct pattern.
    
    The agent iteratively decides which tools to use based on:
    - Query characteristics
    - Current results
    - Previous actions
    
    Sub-queries are processed IN PARALLEL for better performance.
    """
    # Get sub-queries from state (if query was decomposed)
    sub_queries = state.get("sub_queries") or [query]
    target_collections = state.get("target_collections") or ["bukhari", "muslim"]
    query_language = state.get("language", "ar")  # Get detected language
    desired_output_language = state.get("desired_output_language")  # User's preferred result language
    
    metadata["sub_queries"] = sub_queries
    metadata["target_collections"] = target_collections
    metadata["query_language"] = query_language
    metadata["desired_output_language"] = desired_output_language
    
    # Get query_intent for special handling of specific_lookup
    query_intent = state.get("query_intent", "thematic_search")
    metadata["query_intent"] = query_intent
    
    # Process sub-queries in PARALLEL using asyncio
    logger.info(f"Processing {len(sub_queries)} sub-queries in parallel (lang={query_language}, output={desired_output_language}, intent={query_intent})")
    
    all_results = asyncio.run(
        _execute_parallel_agents(
            sub_queries=sub_queries,
            target_collections=target_collections,
            query_language=query_language,
            desired_output_language=desired_output_language,
            query_intent=query_intent,
            metadata=metadata,
        )
    )
    
    # Aggregate results PER sub-query first, then merge
    # This ensures each sub-query contributes equally to the final results
    # Flow: Each sub-query's 50 docs → Rerank against that sub-query → Top 5 → Merge all
    num_sub_queries = len(sub_queries)
    per_query_top_k = DEFAULT_TOP_K  # 5 results per sub-query
    
    logger.info(f"Aggregating results: {num_sub_queries} sub-queries × {per_query_top_k} results each")
    
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
        
        # Aggregate this sub-query's results against its own query (not the original)
        sub_aggregated = aggregate_results(
            raw_results=[docs],  # Single list for this sub-query
            original_query=sub_query,  # Rerank against THIS sub-query
            top_k=per_query_top_k,
            use_reranker=True,
        )
        
        aggregation_stats["per_query_results"].append({
            "sub_query": sub_query[:100],
            "raw_count": len(docs),
            "after_rerank": len(sub_aggregated.documents),
        })
        aggregation_stats["total_unique"] += sub_aggregated.total_unique
        aggregation_stats["duplicates_removed"] += sub_aggregated.duplicates_removed
        aggregation_stats["reranking_applied"] = aggregation_stats["reranking_applied"] or sub_aggregated.reranking_applied
        
        merged_documents.extend(sub_aggregated.documents)
        logger.info(f"Sub-query {i}: {len(docs)} raw → {len(sub_aggregated.documents)} after rerank")
    
    # Final deduplication across all sub-queries (in case same hadith appeared in multiple)
    seen_ids = set()
    final_documents: List[Document] = []
    for doc in merged_documents:
        doc_key = doc.parent_hadith_id or doc.chunk_id
        if doc_key not in seen_ids:
            seen_ids.add(doc_key)
            final_documents.append(doc)
    
    cross_query_duplicates = len(merged_documents) - len(final_documents)
    aggregation_stats["cross_query_duplicates_removed"] = cross_query_duplicates
    
    logger.info(f"Final merge: {len(merged_documents)} → {len(final_documents)} (removed {cross_query_duplicates} cross-query duplicates)")
    
    metadata["aggregation"] = aggregation_stats
    
    # Reassemble chunked hadiths into their complete form
    reassembled_docs = _reassemble_chunked_hadiths(
        documents=final_documents,
        desired_language=desired_output_language,
    )
    
    metadata["reassembly"] = {
        "input_count": len(final_documents),
        "output_count": len(reassembled_docs),
    }
    
    return reassembled_docs


async def _execute_parallel_agents(
    sub_queries: List[str],
    target_collections: List[str],
    query_language: str,
    desired_output_language: Optional[str],
    query_intent: str,
    metadata: Dict[str, Any],
) -> List[List[Document]]:
    """
    Execute autonomous agents for all sub-queries in parallel.
    
    Each sub-query gets its own ReAct agent that runs independently.
    """
    tasks = [
        _execute_autonomous_agent_async(
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
    processed_results = []
    all_iteration_logs = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Agent for sub-query {i} failed: {result}")
            processed_results.append([])
        else:
            docs, iteration_logs = result
            processed_results.append(docs)
            all_iteration_logs.extend(iteration_logs)
    
    # Store all iteration logs in metadata
    metadata["agent_iterations"] = all_iteration_logs
    
    return processed_results


async def _execute_autonomous_agent_async(
    query: str,
    target_collections: List[str],
    query_language: str = "ar",
    desired_output_language: Optional[str] = None,
    query_intent: Optional[str] = None,
    query_index: int = 0,
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    Async version of autonomous agent execution for parallel processing.
    
    Args:
        query: The search query
        target_collections: Collections to search (bukhari, muslim)
        query_language: Detected language of the query (ar/en/mixed)
        desired_output_language: User's explicit preference for result language (arabic/english/None)
        query_intent: Intent of the query (specific_lookup, thematic_search, etc.) - None means unknown
        query_index: Index of this sub-query (for logging)
    
    Returns: (documents, iteration_logs)
    """
    # Agent state
    results: List[Document] = []
    
    # Initialize filters with collection from query analysis
    # Language filter applied ONLY if user explicitly requested a specific language
    filters: Optional[MetadataFilter] = MetadataFilter()
    if len(target_collections) == 1:
        filters.collection = target_collections[0]
        logger.info(f"[Query {query_index}] Filter collection: {target_collections[0]}")
    
    # Apply language filter ONLY if user explicitly requested it
    if desired_output_language:
        filters.language = desired_output_language
        logger.info(f"[Query {query_index}] Filter language: {desired_output_language} (user preference)")
    
    # For specific_lookup intent, pre-extract hadith_id_in_book and other filters from query
    # This ensures that queries like "حديث رقم 70" correctly filter by hadith_id_in_book
    if query_intent == "specific_lookup":
        logger.info(f"[Query {query_index}] Specific lookup detected - extracting metadata filters")
        extracted_filters = extract_metadata_filters(query, use_llm=False)
        
        # Merge extracted filters with existing filters
        # Use hadith_id_in_book (user-facing number) not internal hadith_id
        if extracted_filters.hadith_id_in_book is not None:
            filters.hadith_id_in_book = extracted_filters.hadith_id_in_book
            logger.info(f"[Query {query_index}] Pre-extracted hadith_id_in_book: {filters.hadith_id_in_book}")
        if extracted_filters.book_id is not None:
            filters.book_id = extracted_filters.book_id
            logger.info(f"[Query {query_index}] Pre-extracted book_id: {filters.book_id}")
        if extracted_filters.chapter_id is not None:
            filters.chapter_id = extracted_filters.chapter_id
            logger.info(f"[Query {query_index}] Pre-extracted chapter_id: {filters.chapter_id}")
        if extracted_filters.narrator:
            filters.narrator = extracted_filters.narrator
            logger.info(f"[Query {query_index}] Pre-extracted narrator: {filters.narrator}")
        
        # If we have a hadith_id_in_book, skip the agent loop and do direct lookup
        if filters.hadith_id_in_book is not None:
            logger.info(f"[Query {query_index}] Direct lookup for hadith_id_in_book={filters.hadith_id_in_book}")
            direct_results = await asyncio.to_thread(
                _agent_semantic_search, query, filters, 10
            )
            iteration_logs = [{
                "query_index": query_index,
                "iteration": 1,
                "thought": "Direct lookup for specific hadith number in book",
                "action": "semantic_search",
                "action_input": {"query": query, "hadith_id_in_book": filters.hadith_id_in_book},
                "result": f"Found {len(direct_results)} results for hadith #{filters.hadith_id_in_book}",
            }]
            return (direct_results, iteration_logs)
    
    expanded_terms: List[str] = []
    attempts = 0
    last_result = "Starting search"
    iteration_logs = []
    chapter_found = False  # Track if find_chapter has been called
    
    # ReAct loop
    for iteration in range(MAX_AGENT_ITERATIONS):
        logger.info(f"[Query {query_index}] Agent iteration {iteration + 1}/{MAX_AGENT_ITERATIONS}")
        
        # Get agent decision (run in thread pool to avoid blocking)
        action, action_input, thought = await asyncio.to_thread(
            _get_agent_decision,
            query=query,
            sub_queries=[query],
            results_count=len(results),
            attempts=attempts,
            last_result=last_result,
        )
        
        iteration_log = {
            "query_index": query_index,
            "iteration": iteration + 1,
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }
        
        logger.info(f"[Query {query_index}] Agent action: {action}")
        
        # Execute action (in thread pool for sync functions)
        if action == "expand_query":
            expanded_terms, last_result = await asyncio.to_thread(
                _agent_expand_query, query
            )
            iteration_log["result"] = last_result
            
        elif action == "extract_filters":
            # Extract new filters but PRESERVE the language filter
            existing_language = filters.language if filters else None
            new_filters, last_result = await asyncio.to_thread(
                _agent_extract_filters, query, target_collections
            )
            if new_filters:
                # Merge: keep existing language filter if new_filters doesn't have one
                if existing_language and not new_filters.language:
                    new_filters.language = existing_language
                filters = new_filters
            iteration_log["result"] = last_result
            
            # Auto-execute semantic search if we have a hadith_id filter (specific lookup)
            if filters and filters.hadith_id:
                logger.info(f"[Query {query_index}] Auto-search: hadith_id filter detected")
                new_results = await asyncio.to_thread(
                    _agent_semantic_search, query, filters, PARALLEL_SEARCH_K
                )
                results.extend(new_results)
                attempts += 1
                last_result = f"Found {len(new_results)} results for hadith #{filters.hadith_id}"
                iteration_log["result"] = last_result
            
        elif action == "find_chapter":
            # Prevent repeated find_chapter calls - force hybrid_search instead
            if chapter_found:
                logger.info(f"[Query {query_index}] Chapter already found, forcing hybrid_search")
                action = "hybrid_search"
                k = action_input.get("k", PARALLEL_SEARCH_K)
                new_results = await asyncio.to_thread(
                    _agent_hybrid_search, query, filters, k
                )
                results.extend(new_results)
                attempts += 1
                last_result = f"Found {len(new_results)} results (auto-search after chapter found)"
                iteration_log["action"] = "hybrid_search (auto)"
                iteration_log["result"] = last_result
            else:
                subject = action_input.get("subject", query)
                coll = action_input.get("collection")
                if not coll and len(target_collections) == 1:
                    coll = target_collections[0]
                chapter_id, last_result = await asyncio.to_thread(
                    _agent_find_chapter, subject, coll
                )
                if chapter_id:
                    # Update or create filters with found chapter_id
                    if filters is None:
                        filters = MetadataFilter()
                    filters.chapter_id = chapter_id
                    chapter_found = True
                iteration_log["result"] = last_result
            
        elif action == "keyword_search":
            # Always use PARALLEL_SEARCH_K for initial retrieval, cross-encoder will filter to top results
            k = PARALLEL_SEARCH_K
            # Priority: 1. Agent's refined query 2. Expanded terms 3. Original query
            if "query" in action_input and action_input["query"]:
                search_query = action_input["query"]
            elif len(expanded_terms) >= 2:
                search_query = ' '.join(expanded_terms[:4])
            else:
                search_query = query
                
            new_results = await asyncio.to_thread(
                _agent_keyword_search, search_query, filters, k
            )
            results.extend(new_results)
            attempts += 1
            last_result = f"Found {len(new_results)} results"
            iteration_log["result"] = last_result
            
        elif action == "semantic_search":
            # Always use PARALLEL_SEARCH_K for initial retrieval, cross-encoder will filter to top results
            k = PARALLEL_SEARCH_K
            # Priority: 1. Agent's refined query 2. Expanded terms 3. Original query
            if "query" in action_input and action_input["query"]:
                search_query = action_input["query"]
            elif len(expanded_terms) >= 2:
                search_query = ' '.join(expanded_terms[:4])
            else:
                search_query = query
                
            new_results = await asyncio.to_thread(
                _agent_semantic_search, search_query, filters, k
            )
            results.extend(new_results)
            attempts += 1
            last_result = f"Found {len(new_results)} results"
            iteration_log["result"] = last_result
            
        elif action == "hybrid_search":
            # Always use PARALLEL_SEARCH_K for initial retrieval, cross-encoder will filter to top results
            k = PARALLEL_SEARCH_K
            # Priority: 1. Agent's refined query 2. Expanded terms 3. Original query
            if "query" in action_input and action_input["query"]:
                search_query = action_input["query"]
            elif len(expanded_terms) >= 2:
                search_query = ' '.join(expanded_terms[:4])
            else:
                search_query = query
                
            new_results = await asyncio.to_thread(
                _agent_hybrid_search, search_query, filters, k
            )
            results.extend(new_results)
            attempts += 1
            last_result = f"Found {len(new_results)} results"
            iteration_log["result"] = last_result
            
        elif action == "relax_filters":
            level = action_input.get("level", 1)
            if filters:
                filters = filters.relax(level=level)
                last_result = f"Relaxed filters to level {level}"
            else:
                last_result = "No filters to relax"
            iteration_log["result"] = last_result
            
        elif action == "finish":
            reason = action_input.get("reason", "Agent decided to finish")
            last_result = reason
            iteration_log["result"] = reason
            iteration_logs.append(iteration_log)
            break
            
        else:
            logger.warning(f"[Query {query_index}] Unknown action: {action}, defaulting to hybrid_search")
            new_results = await asyncio.to_thread(
                _agent_hybrid_search, query, filters, PARALLEL_SEARCH_K
            )
            results.extend(new_results)
            attempts += 1
            last_result = f"Found {len(new_results)} results (fallback)"
            iteration_log["result"] = last_result
        
        iteration_logs.append(iteration_log)
        
        # Auto-finish conditions
        if len(results) >= 5 and attempts >= 1:
            logger.info(f"[Query {query_index}] Auto-finish: Sufficient results found")
            break
        
        if attempts >= 3 and len(results) == 0:
            logger.info(f"[Query {query_index}] Auto-finish: Max attempts with no results")
            break
    
    # Deduplicate results
    seen_ids = set()
    unique_results = []
    for doc in results:
        if doc.chunk_id not in seen_ids:
            seen_ids.add(doc.chunk_id)
            unique_results.append(doc)
    
    return unique_results, iteration_logs


@traceable(name="agent_iteration", run_type="chain")
def _execute_autonomous_agent(
    query: str,
    target_collections: List[str],
    metadata: Dict[str, Any],
) -> List[Document]:
    """
    Execute the ReAct agent loop for a single query.
    
    The agent decides at each step:
    1. expand_query - Should I expand with synonyms?
    2. extract_filters - Should I extract metadata filters?
    3. keyword_search / semantic_search / hybrid_search - Which tool?
    4. relax_filters - Should I broaden the search?
    5. finish - Have I found enough results?
    """
    # Agent state
    results: List[Document] = []
    
    # Initialize filters with collection from target_collections
    # This ensures collection filtering is always applied
    filters: Optional[MetadataFilter] = None
    if len(target_collections) == 1:
        filters = MetadataFilter(collection=target_collections[0])
        logger.info(f"Initialized filter with collection: {target_collections[0]}")
    
    expanded_terms: List[str] = []
    attempts = 0
    last_result = "Starting search"
    iteration_logs = []
    chapter_found = False  # Track if find_chapter has been called
    
    # ReAct loop
    for iteration in range(MAX_AGENT_ITERATIONS):
        logger.info(f"Agent iteration {iteration + 1}/{MAX_AGENT_ITERATIONS}")
        
        # Get agent decision
        action, action_input, thought = _get_agent_decision(
            query=query,
            sub_queries=[query],
            results_count=len(results),
            attempts=attempts,
            last_result=last_result,
        )
        
        iteration_log = {
            "iteration": iteration + 1,
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }
        
        logger.info(f"Agent thought: {thought}")
        logger.info(f"Agent action: {action}")
        
        # Execute action
        if action == "expand_query":
            expanded_terms, last_result = _agent_expand_query(query)
            iteration_log["result"] = last_result
            
        elif action == "extract_filters":
            filters, last_result = _agent_extract_filters(query, target_collections)
            iteration_log["result"] = last_result
            
        elif action == "find_chapter":
            # Prevent repeated find_chapter calls - force hybrid_search instead
            if chapter_found:
                logger.info(f"Chapter already found, forcing hybrid_search")
                action = "hybrid_search"
                k = action_input.get("k", PARALLEL_SEARCH_K)
                new_results = _agent_hybrid_search(query, filters, k)
                results.extend(new_results)
                attempts += 1
                last_result = f"Found {len(new_results)} results (auto-search after chapter found)"
                iteration_log["action"] = "hybrid_search (auto)"
                iteration_log["result"] = last_result
            else:
                subject = action_input.get("subject", query)
                coll = action_input.get("collection")
                if not coll and len(target_collections) == 1:
                    coll = target_collections[0]
                chapter_id, last_result = _agent_find_chapter(subject, coll)
                if chapter_id:
                    # Update or create filters with found chapter_id
                    if filters is None:
                        filters = MetadataFilter()
                    filters.chapter_id = chapter_id
                    chapter_found = True
                iteration_log["result"] = last_result
            
        elif action == "keyword_search":
            k = action_input.get("k", PARALLEL_SEARCH_K)
            new_results = _agent_keyword_search(query, filters, k)
            results.extend(new_results)
            attempts += 1
            last_result = f"Found {len(new_results)} results"
            iteration_log["result"] = last_result
            
        elif action == "semantic_search":
            k = action_input.get("k", PARALLEL_SEARCH_K)
            new_results = _agent_semantic_search(query, filters, k)
            results.extend(new_results)
            attempts += 1
            last_result = f"Found {len(new_results)} results"
            iteration_log["result"] = last_result
            
        elif action == "hybrid_search":
            k = action_input.get("k", PARALLEL_SEARCH_K)
            new_results = _agent_hybrid_search(query, filters, k)
            results.extend(new_results)
            attempts += 1
            last_result = f"Found {len(new_results)} results"
            iteration_log["result"] = last_result
            
        elif action == "relax_filters":
            level = action_input.get("level", 1)
            if filters:
                filters = filters.relax(level=level)
                last_result = f"Relaxed filters to level {level}"
            else:
                last_result = "No filters to relax"
            iteration_log["result"] = last_result
            
        elif action == "finish":
            reason = action_input.get("reason", "Agent decided to finish")
            last_result = reason
            iteration_log["result"] = reason
            iteration_logs.append(iteration_log)
            break
            
        else:
            logger.warning(f"Unknown action: {action}, defaulting to hybrid_search")
            new_results = _agent_hybrid_search(query, filters, PARALLEL_SEARCH_K)
            results.extend(new_results)
            attempts += 1
            last_result = f"Found {len(new_results)} results (fallback)"
            iteration_log["result"] = last_result
        
        iteration_logs.append(iteration_log)
        
        # Auto-finish conditions
        if len(results) >= 5 and attempts >= 1:
            logger.info("Auto-finish: Sufficient results found")
            break
        
        if attempts >= 3 and len(results) == 0:
            logger.info("Auto-finish: Max attempts with no results")
            break
    
    metadata["agent_iterations"].extend(iteration_logs)
    
    # Deduplicate results
    seen_ids = set()
    unique_results = []
    for doc in results:
        if doc.chunk_id not in seen_ids:
            seen_ids.add(doc.chunk_id)
            unique_results.append(doc)
    
    return unique_results


@traceable(name="agent_decision", run_type="chain")
def _get_agent_decision(
    query: str,
    sub_queries: List[str],
    results_count: int,
    attempts: int,
    last_result: str,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Get the agent's next action using LLM.
    
    Uses call_llm_sync to avoid nested event loop issues when called
    from asyncio.to_thread in parallel execution.
    
    Returns: (action_name, action_input, thought)
    """
    system_message, prompt, temperature, max_tokens = format_prompt(
        "retrieval", "autonomous_agent",
        query=query,
        sub_queries=str(sub_queries),
        results_count=results_count,
        attempts=attempts,
        last_result=last_result,
    )
    
    try:
        response = call_llm_sync(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata={"tool": "autonomous_agent"}
        )
        
        parsed = parse_json_response(response)
        
        action = parsed.get("action", "hybrid_search")
        action_input = parsed.get("action_input", {})
        thought = parsed.get("thought", "No reasoning provided")
        
        # Validate action_input is a dict
        if not isinstance(action_input, dict):
            action_input = {}
        
        return action, action_input, thought
        
    except Exception as e:
        logger.warning(f"Agent decision failed: {e}, defaulting to hybrid_search")
        return "hybrid_search", {"k": PARALLEL_SEARCH_K}, f"Error: {e}, using hybrid_search"


# ============================================================================
# Agent Tool Implementations (Traced)
# ============================================================================

@traceable(name="agent_tool_find_chapter", run_type="tool")
def _agent_find_chapter(
    subject: str,
    collection: Optional[str] = None,
) -> Tuple[Optional[int], str]:
    """
    Agent tool: Find chapter ID for a subject term.
    
    Maps user subject terms (e.g., "البيع", "sales") to chapter IDs
    for precise filtering in subsequent searches.
    """
    try:
        result = find_chapter_for_subject(subject, collection)
        if result:
            return (
                result.chapter_id,
                f"Chapter filter applied: chapter_id={result.chapter_id} ({result.chapter_title_en} / {result.chapter_title_ar}). "
                f"Confidence: {result.confidence:.2f}. Ready for search."
            )
        return None, f"No chapter found for '{subject}'. No filter applied."
    except Exception as e:
        logger.warning(f"Chapter lookup failed: {e}")
        return None, f"Chapter lookup error: {e}. No filter applied."


@traceable(name="agent_tool_expand_query", run_type="tool")
def _agent_expand_query(query: str) -> Tuple[List[str], str]:
    """Agent tool: Expand query with synonyms/translations."""
    try:
        expander = QueryExpansionTool()
        result = expander(query, use_llm=True)
        terms = result.expanded_terms
        return terms, f"Expanded to {len(terms)} terms: {terms[:3]}..."
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return [], f"Expansion failed: {e}"


@traceable(name="agent_tool_extract_filters", run_type="tool")
def _agent_extract_filters(
    query: str,
    target_collections: List[str],
) -> Tuple[Optional[MetadataFilter], str]:
    """Agent tool: Extract metadata filters from query."""
    try:
        filters = extract_metadata_filters(query, use_llm=True)
        
        # Set collection filter based on targets
        if len(target_collections) == 1:
            filters.collection = target_collections[0]
        
        filter_dict = filters.model_dump(exclude_none=True)
        return filters, f"Extracted filters: {filter_dict}"
    except Exception as e:
        logger.warning(f"Filter extraction failed: {e}")
        return None, f"Filter extraction failed: {e}"


@traceable(name="agent_tool_keyword_search", run_type="retriever")
def _agent_keyword_search(
    query: str,
    filters: Optional[MetadataFilter],
    k: int,
) -> List[Document]:
    """Agent tool: Execute keyword (BM25) search."""
    try:
        result = keyword_search(
            query=query,
            k=k,
            filters=filters if filters and not filters.is_empty() else None,
        )
        return result.documents
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return []


@traceable(name="agent_tool_semantic_search", run_type="retriever")
def _agent_semantic_search(
    query: str,
    filters: Optional[MetadataFilter],
    k: int,
) -> List[Document]:
    """Agent tool: Execute semantic (vector) search."""
    try:
        result = semantic_search(
            query=query,
            k=k,
            filters=filters if filters and not filters.is_empty() else None,
        )
        return result.documents
    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
        return []


@traceable(name="agent_tool_hybrid_search", run_type="retriever")
def _agent_hybrid_search(
    query: str,
    filters: Optional[MetadataFilter],
    k: int,
) -> List[Document]:
    """
    Agent tool: Execute hybrid (combined) search.
    
    **v3.0 Enhancement**: Uses HybridSearchTool which automatically
    detects Arabic queries and applies cross-lingual search strategy:
    - BM25 keyword search on Arabic keywords
    - Vector search with translated English query
    - Vector search on Arabic docs
    - RRF fusion of all results
    """
    try:
        # Use HybridSearchTool which has cross-lingual support for Arabic
        tool = HybridSearchTool(use_crosslingual=True)
        result = tool(
            query=query,
            k=k,
            alpha=0.6,  # Slightly prefer semantic
            filters=filters if filters and not filters.is_empty() else None,
        )
        return result.documents
    except Exception as e:
        logger.warning(f"Hybrid search failed: {e}")
        return []


def _handle_user_text(
    query: str,
    state: AgentState,
    metadata: Dict[str, Any],
) -> List[Document]:
    """
    Handle user-provided text by finding similar authentic hadiths.
    
    Routes to UserHadithProcessorTool.
    """
    logger.info("Routing to user text processor")
    metadata["stages"].append("user_text_processing")
    
    try:
        processor = UserHadithProcessorTool()
        result = processor(user_text=query, find_similar=True, top_k=DEFAULT_TOP_K)
        
        metadata["user_text_result"] = {
            "is_authentic": result.is_authentic,
            "match_confidence": result.match_confidence,
            "similar_found": len(result.similar_hadiths),
        }
        
        return result.similar_hadiths
        
    except Exception as e:
        logger.error(f"User text processing failed: {e}")
        metadata["errors"].append({"stage": "user_text", "error": str(e)})
        return []


@traceable(name="handle_metadata_query", run_type="chain")
def _handle_metadata_query(
    query: str,
    state: AgentState,
    metadata: Dict[str, Any],
) -> List[Document]:
    """
    Handle metadata-based queries (longest, shortest, most, count, etc.)
    
    These queries require direct database lookups rather than semantic search.
    Uses the total_chunks metadata field to determine hadith length:
    - More chunks = longer hadith (each chunk is ~800 chars)
    
    Supported query types:
    - "longest hadith" -> find hadith with max total_chunks
    - "shortest hadith" -> find hadith with min total_chunks (=1)
    - "how many hadiths" -> count query (future)
    """
    from src.utils.singletons import get_chroma_client
    
    logger.info("Handling metadata query")
    metadata["stages"].append("metadata_query")
    
    target_collections = state.get("target_collections") or ["bukhari", "muslim"]
    desired_language = state.get("desired_output_language")
    
    # Determine query type from the query text
    query_lower = query.lower()
    is_longest = any(term in query_lower for term in ['أطول', 'اطول', 'longest', 'long'])
    is_shortest = any(term in query_lower for term in ['أقصر', 'اقصر', 'shortest', 'short'])
    
    # If neither longest nor shortest detected, this is NOT a metadata query
    # Fall back to autonomous search (e.g., "ما عدد الصلوات" is asking about content, not hadith stats)
    if not is_longest and not is_shortest:
        logger.info("Metadata query type not detected (not longest/shortest), falling back to autonomous search")
        metadata["stages"].append("metadata_fallback_to_search")
        return _autonomous_search(query, state, metadata)
    
    try:
        client = get_chroma_client()
        results = []
        
        for coll_name in target_collections:
            collection_db_name = f"hadith_{coll_name}"
            try:
                collection = client.get_collection(collection_db_name)
            except Exception as e:
                logger.warning(f"Collection {collection_db_name} not found: {e}")
                continue
            
            # Get all documents with metadata to find longest/shortest
            all_docs = collection.get(include=['metadatas', 'documents'])
            
            if not all_docs['ids']:
                continue
            
            # Build hadith -> chunks mapping with text length tracking
            hadith_chunks = {}  # hadith_id -> data dict
            for i, (doc_id, meta, text) in enumerate(zip(
                all_docs['ids'], 
                all_docs['metadatas'], 
                all_docs['documents']
            )):
                hadith_id = meta.get('hadith_id')
                total_chunks = meta.get('total_chunks', 1)
                lang = meta.get('language', 'arabic')
                
                # Filter by desired language if specified
                if desired_language:
                    if desired_language == 'arabic' and lang != 'arabic':
                        continue
                    if desired_language == 'english' and lang != 'english':
                        continue
                
                # Track chunks per hadith
                if hadith_id not in hadith_chunks:
                    hadith_chunks[hadith_id] = {
                        'total_chunks': total_chunks,
                        'doc_ids': [doc_id],
                        'metadatas': [meta],
                        'texts': [text],
                        'total_text_length': len(text),  # Track actual text length
                    }
                else:
                    # Add chunk to existing hadith
                    hadith_chunks[hadith_id]['doc_ids'].append(doc_id)
                    hadith_chunks[hadith_id]['metadatas'].append(meta)
                    hadith_chunks[hadith_id]['texts'].append(text)
                    hadith_chunks[hadith_id]['total_text_length'] += len(text)
                    # Update total_chunks if this chunk has higher value
                    if total_chunks > hadith_chunks[hadith_id]['total_chunks']:
                        hadith_chunks[hadith_id]['total_chunks'] = total_chunks
            
            if not hadith_chunks:
                continue
            
            # Sort hadiths based on query type
            if is_shortest:
                # For shortest: sort by actual text length (ascending)
                sorted_hadiths = sorted(
                    hadith_chunks.items(),
                    key=lambda x: x[1]['total_text_length'],
                    reverse=False  # Ascending - shortest first
                )
            else:
                # For longest: sort by total_chunks (descending) then by text length
                sorted_hadiths = sorted(
                    hadith_chunks.items(),
                    key=lambda x: (x[1]['total_chunks'], x[1]['total_text_length']),
                    reverse=True  # Descending - longest first
                )
            
            # Get top result
            if sorted_hadiths:
                top_hadith_id, top_data = sorted_hadiths[0]
                
                # For metadata queries, we need ALL chunks to reconstruct the full hadith
                # Query ChromaDB again to get all chunks for this specific hadith
                all_hadith_chunks = collection.get(
                    where={
                        "$and": [
                            {"hadith_id": {"$eq": top_hadith_id}},
                            {"language": {"$eq": desired_language if desired_language else "arabic"}}
                        ]
                    },
                    include=['metadatas', 'documents']
                )
                
                # Sort chunks by chunk_index to reconstruct in order
                chunk_data = list(zip(
                    all_hadith_chunks['ids'],
                    all_hadith_chunks['metadatas'],
                    all_hadith_chunks['documents']
                ))
                chunk_data.sort(key=lambda x: x[1].get('chunk_index', 0))
                
                # Combine all chunk texts into one
                combined_text = "\n".join([text for _, _, text in chunk_data])
                
                # Use metadata from first chunk
                first_meta = chunk_data[0][1] if chunk_data else top_data['metadatas'][0]
                first_doc_id = chunk_data[0][0] if chunk_data else top_data['doc_ids'][0]
                
                doc = Document(
                    chunk_id=first_doc_id,
                    text=combined_text,  # Full combined text from all chunks
                    score=1.0,  # Top result
                    search_type="metadata_query",
                    language=first_meta.get('language', 'arabic'),
                    collection=first_meta.get('collection', ''),
                    book_id=first_meta.get('book_id'),
                    chapter_id=first_meta.get('chapter_id'),
                    hadith_id=first_meta.get('hadith_id'),
                    narrator=first_meta.get('narrator'),
                    parent_hadith_id=first_meta.get('parent_hadith_id'),
                    book_number=first_meta.get('book_number'),
                    chapter_number=first_meta.get('chapter_number'),
                    hadith_id_in_book=first_meta.get('hadith_id_in_book'),
                    chunk_index=0,  # Combined document
                    total_chunks=first_meta.get('total_chunks', 1),
                    is_chunked=first_meta.get('is_chunked', False),
                )
                results.append(doc)
                
                metadata["metadata_query_result"] = {
                    "query_type": "longest" if is_longest else "shortest",
                    "hadith_id": top_hadith_id,
                    "total_chunks": top_data['total_chunks'],
                    "collection": coll_name,
                }
                
                logger.info(
                    f"Metadata query found: Hadith #{top_hadith_id} with "
                    f"{top_data['total_chunks']} chunks ({coll_name})"
                )
        
        return results
        
    except Exception as e:
        logger.error(f"Metadata query failed: {e}")
        metadata["errors"].append({"stage": "metadata_query", "error": str(e)})
        # Fall back to semantic search
        logger.info("Falling back to semantic search")
        return _autonomous_search(query, state, metadata)


def _handle_base_knowledge_search(
    query: str,
    state: AgentState,
    metadata: Dict[str, Any],
) -> List[Document]:
    """
    Handle search against the hadith knowledge base.
    
    Uses the autonomous ReAct agent to dynamically decide:
    - Which search tools to use (keyword, semantic, hybrid)
    - Whether to expand the query
    - Whether to extract metadata filters
    - When to relax filters and retry
    """
    logger.info("Executing autonomous ReAct retrieval agent")
    
    # Get sub-queries from state (if query was decomposed)
    sub_queries = state.get("sub_queries") or [query]
    
    # Get target collections
    target_collections = state.get("target_collections") or ["bukhari", "muslim"]
    
    metadata["sub_queries"] = sub_queries
    metadata["target_collections"] = target_collections
    metadata["stages"].append("react_agent")
    
    # Run the ReAct agent
    results = _run_react_agent(
        query=query,
        sub_queries=sub_queries,
        target_collections=target_collections,
        metadata=metadata,
    )
    
    # Aggregate if we have results from multiple sub-queries
    if len(results) > DEFAULT_TOP_K:
        aggregated = aggregate_results(
            raw_results=[results],
            original_query=query,
            top_k=DEFAULT_TOP_K,
            use_reranker=True,
        )
        return aggregated.documents
    
    return results[:DEFAULT_TOP_K]


def _expand_all_queries(queries: List[str]) -> Dict[str, List[str]]:
    """
    Expand all queries with synonyms and translations.
    
    Returns dict mapping original query to expanded terms.
    """
    expansions = {}
    expander = QueryExpansionTool()
    
    for query in queries:
        try:
            expanded = expander(query, use_llm=False)  # Fast dictionary-based
            expansions[query] = expanded.expanded_terms
        except Exception as e:
            logger.warning(f"Query expansion failed for '{query[:50]}...': {e}")
            expansions[query] = []
    
    return expansions


# ============================================================================
# ReAct Agent Loop
# ============================================================================

# Constants for ReAct agent
MAX_AGENT_ITERATIONS = 5
MIN_RESULTS_TO_FINISH = 5

REACT_TOOL_DEFINITIONS = {
    "expand_query": "Generate synonyms/translations to improve recall",
    "extract_filters": "Extract metadata filters (collection, chapter, hadith_id, narrator)",
    "keyword_search": "BM25 lexical search for exact term matching",
    "semantic_search": "Vector similarity search for conceptual matching",
    "hybrid_search": "Combined search with RRF (best for general queries)",
    "relax_filters": "Remove strict filters to broaden search",
    "finish": "Return final results",
}


@traceable(name="react_agent_loop", run_type="chain")
def _run_react_agent(
    query: str,
    sub_queries: List[str],
    target_collections: List[str],
    metadata: Dict[str, Any],
) -> List[Document]:
    """
    Run autonomous ReAct agent loop for retrieval.
    
    The agent decides which tools to use based on:
    1. Query characteristics
    2. Current results count
    3. Search attempts made
    4. Last action outcome
    
    Args:
        query: Original query
        sub_queries: Decomposed sub-queries (if any)
        target_collections: Target collections to search
        metadata: Metadata dict to update with agent trace
        
    Returns:
        List of retrieved documents
    """
    logger.info(f"Starting ReAct agent for query: {query[:50]}...")
    
    # Agent state
    results: List[Document] = []
    filters: Optional[MetadataFilter] = None
    expanded_terms: List[str] = []
    search_attempts = 0
    agent_trace: List[Dict[str, Any]] = []
    filter_relaxation_level = 0
    
    # Process sub-queries (use original if no decomposition)
    search_query = sub_queries[0] if sub_queries else query
    
    for iteration in range(MAX_AGENT_ITERATIONS):
        logger.info(f"ReAct iteration {iteration + 1}/{MAX_AGENT_ITERATIONS}")
        
        # Prepare context for agent decision
        last_result = "No actions taken yet" if iteration == 0 else agent_trace[-1].get("result", "Unknown")
        
        # Call LLM for next action
        try:
            action_decision = _get_agent_action(
                query=search_query,
                sub_queries=sub_queries,
                results_count=len(results),
                attempts=search_attempts,
                last_result=last_result,
            )
        except Exception as e:
            logger.error(f"Agent decision failed: {e}")
            # Fallback to hybrid search
            action_decision = {
                "thought": f"Decision failed: {e}, using fallback",
                "action": "hybrid_search",
                "action_input": {"query": search_query, "k": DEFAULT_TOP_K},
            }
        
        thought = action_decision.get("thought", "")
        action = action_decision.get("action", "finish")
        action_input = action_decision.get("action_input", {})
        
        logger.info(f"Agent action: {action} | Thought: {thought[:100]}...")
        
        # Record trace
        trace_entry = {
            "iteration": iteration + 1,
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }
        
        # Execute action
        try:
            if action == "finish":
                trace_entry["result"] = f"Finishing with {len(results)} results"
                agent_trace.append(trace_entry)
                logger.info(f"Agent finished: {action_input.get('reason', 'No reason given')}")
                break
                
            elif action == "expand_query":
                input_query = action_input.get("query", search_query)
                expander = QueryExpansionTool()
                expansion_result = expander(input_query, use_llm=False)
                expanded_terms = expansion_result.expanded_terms
                trace_entry["result"] = f"Expanded to {len(expanded_terms)} terms: {expanded_terms[:3]}"
                
            elif action == "extract_filters":
                input_query = action_input.get("query", search_query)
                filters = extract_metadata_filters(input_query, use_llm=True)
                # Set collection filter
                if target_collections and len(target_collections) == 1:
                    filters.collection = target_collections[0]
                trace_entry["result"] = f"Extracted filters: {filters.model_dump(exclude_none=True)}"
                
            elif action == "keyword_search":
                search_attempts += 1
                search_query_input = action_input.get("query", search_query)
                k = action_input.get("k", DEFAULT_TOP_K)
                current_filters = filters.relax(level=filter_relaxation_level) if filters else None
                
                result = keyword_search(
                    query=search_query_input,
                    k=k,
                    filters=current_filters if current_filters and not current_filters.is_empty() else None,
                )
                new_docs = result.documents
                results = _merge_results(results, new_docs)
                trace_entry["result"] = f"Found {len(new_docs)} docs, total now {len(results)}"
                
            elif action == "semantic_search":
                search_attempts += 1
                search_query_input = action_input.get("query", search_query)
                k = action_input.get("k", DEFAULT_TOP_K)
                current_filters = filters.relax(level=filter_relaxation_level) if filters else None
                
                result = semantic_search(
                    query=search_query_input,
                    k=k,
                    filters=current_filters if current_filters and not current_filters.is_empty() else None,
                )
                new_docs = result.documents
                results = _merge_results(results, new_docs)
                trace_entry["result"] = f"Found {len(new_docs)} docs, total now {len(results)}"
                
            elif action == "hybrid_search":
                search_attempts += 1
                search_query_input = action_input.get("query", search_query)
                k = action_input.get("k", DEFAULT_TOP_K)
                current_filters = filters.relax(level=filter_relaxation_level) if filters else None
                
                result = hybrid_search(
                    query=search_query_input,
                    k=k,
                    alpha=0.6,
                    filters=current_filters if current_filters and not current_filters.is_empty() else None,
                )
                new_docs = result.documents
                results = _merge_results(results, new_docs)
                trace_entry["result"] = f"Found {len(new_docs)} docs, total now {len(results)}"
                
            elif action == "relax_filters":
                level = action_input.get("level", filter_relaxation_level + 1)
                filter_relaxation_level = min(level, 3)  # Max level 3
                trace_entry["result"] = f"Filters relaxed to level {filter_relaxation_level}"
                
            else:
                trace_entry["result"] = f"Unknown action: {action}"
                logger.warning(f"Unknown agent action: {action}")
                
        except Exception as e:
            trace_entry["result"] = f"Error: {str(e)}"
            logger.error(f"Agent action {action} failed: {e}")
        
        agent_trace.append(trace_entry)
        
        # Auto-finish conditions
        if len(results) >= MIN_RESULTS_TO_FINISH and search_attempts >= 1:
            logger.info(f"Auto-finishing: {len(results)} results found after {search_attempts} searches")
            break
    
    # Store trace in metadata
    metadata["agent_trace"] = agent_trace
    metadata["total_iterations"] = len(agent_trace)
    metadata["search_attempts"] = search_attempts
    
    return results


@traceable(name="agent_decide_action", run_type="chain")
def _get_agent_action(
    query: str,
    sub_queries: List[str],
    results_count: int,
    attempts: int,
    last_result: str,
) -> Dict[str, Any]:
    """
    Get the next action from the LLM agent.
    
    Uses the autonomous_agent prompt to decide:
    - Which tool to use next
    - What inputs to provide
    - When to stop
    """
    system_message, prompt, temperature, max_tokens = format_prompt(
        "retrieval", "autonomous_agent",
        query=query,
        sub_queries=", ".join(sub_queries) if sub_queries else query,
        results_count=results_count,
        attempts=attempts,
        last_result=last_result,
    )
    
    response = call_llm_sync(
        prompt=prompt,
        system_message=system_message,
        temperature=temperature,
        max_tokens=max_tokens,
        metadata={"tool": "react_agent_decision"},
    )
    
    # Parse response
    decision = parse_json_response(response)
    
    # Validate action
    action = decision.get("action", "finish")
    if action not in REACT_TOOL_DEFINITIONS:
        logger.warning(f"Invalid action '{action}', defaulting to hybrid_search")
        decision["action"] = "hybrid_search"
        decision["action_input"] = {"query": query, "k": DEFAULT_TOP_K}
    
    return decision


# ============================================================================
# Async Parallel Execution (Legacy - kept for backward compatibility)
# ============================================================================

async def _execute_parallel_searches(
    sub_queries: List[str],
    expanded_terms: Dict[str, List[str]],
    target_collections: List[str],
    top_k: int = 15,
) -> List[List[Document]]:
    """
    Execute searches for all sub-queries in parallel.
    
    LEGACY: Kept for backward compatibility with retrieve_async.
    New code should use _run_react_agent instead.
    """
    tasks = [
        _execute_search_with_retry(
            query=query,
            expanded_terms=expanded_terms.get(query, []),
            target_collections=target_collections,
            top_k=top_k,
        )
        for query in sub_queries
    ]
    
    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Search task {i} failed: {result}")
            processed_results.append([])
        else:
            processed_results.append(result)
    
    return processed_results


@traceable(name="select_retrieval_tool", run_type="chain")
async def _select_tool(query: str) -> str:
    """
    Select the best retrieval tool for the query using LLM.
    """
    system_message, prompt, temperature, max_tokens = format_prompt(
        "retrieval", "tool_selection", query=query
    )
    
    try:
        response = await call_llm(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata={"tool": "tool_selection"}
        )
        
        parsed = parse_json_response(response)
        tool = parsed.get("tool", "hybrid_search")
        logger.info(f"Selected tool for '{query[:30]}...': {tool} (Reason: {parsed.get('reasoning')})")
        return tool
        
    except Exception as e:
        logger.warning(f"Tool selection failed: {e}, defaulting to hybrid_search")
        return "hybrid_search"


async def _execute_search_with_retry(
    query: str,
    expanded_terms: List[str],
    target_collections: List[str],
    top_k: int = 15,
) -> List[Document]:
    """
    Execute search for a single query with self-correction (retry logic).
    
    Self-Correction Strategy:
    - Attempt 1: Strict filters (all metadata)
    - Attempt 2: Relax filters (remove chapter)
    - Attempt 3: Relax more (remove book)
    - Attempt 4: No filters (collection only)
    
    Args:
        query: Search query
        expanded_terms: Additional search terms
        target_collections: Target collections
        top_k: Number of results
        
    Returns:
        List of retrieved documents
    """
    logger.debug(f"Executing search for: '{query[:50]}...'")
    
    # Extract metadata filters from query
    # Use thread pool for LLM call to avoid blocking event loop
    filters = await asyncio.to_thread(extract_metadata_filters, query, True)
    
    # Set collection filter based on targets
    if len(target_collections) == 1:
        filters.collection = target_collections[0]
    
    # Select tool dynamically
    selected_tool = await _select_tool(query)

    # Try with progressively relaxed filters
    # Reduced retries to avoid excessive latency
    max_attempts = 2 if filters.is_empty() else MAX_RETRIES + 1
    
    for attempt in range(max_attempts):
        # Relax filters based on attempt number
        current_filters = filters.relax(level=attempt) if attempt > 0 else filters
        
        logger.debug(f"Search attempt {attempt + 1}/{max_attempts} with filters: {current_filters.model_dump(exclude_none=True)}")
        
        try:
            # Execute search based on selected tool
            if selected_tool == "keyword_search":
                result = await asyncio.to_thread(
                    keyword_search,
                    query=query,
                    k=top_k,
                    filters=current_filters if not current_filters.is_empty() else None,
                )
            elif selected_tool == "semantic_search":
                result = await asyncio.to_thread(
                    semantic_search,
                    query=query,
                    k=top_k,
                    filters=current_filters if not current_filters.is_empty() else None,
                )
            else:
                # Default to hybrid
                result = await asyncio.to_thread(
                    hybrid_search,
                    query=query,
                    k=top_k,
                    alpha=0.6,  # Slightly prefer semantic
                    filters=current_filters if not current_filters.is_empty() else None,
                )
            
            documents = result.documents
            
            # Check if we got results
            if documents:
                logger.debug(f"Found {len(documents)} documents on attempt {attempt + 1}")
                
                # Also search expanded terms and merge
                # Only do this on first attempt to save time
                if expanded_terms and attempt == 0:
                    expanded_docs = await _search_expanded_terms(
                        expanded_terms[:2],  # Limit expansion searches
                        current_filters,
                        top_k=5,
                    )
                    documents = _merge_results(documents, expanded_docs)
                
                return documents
            
            # No results, try with relaxed filters
            logger.debug(f"No results on attempt {attempt + 1}, relaxing filters")
            
        except Exception as e:
            logger.warning(f"Search attempt {attempt + 1} failed: {e}")
    
    # All attempts exhausted
    logger.warning(f"All search attempts exhausted for query: '{query[:50]}...'")
    return []


async def _search_expanded_terms(
    terms: List[str],
    filters: Optional[MetadataFilter],
    top_k: int = 5,
) -> List[Document]:
    """
    Search for expanded terms and collect results.
    """
    all_docs = []
    
    for term in terms:
        try:
            result = await asyncio.to_thread(
                semantic_search,
                query=term,
                k=top_k,
                filters=filters if filters and not filters.is_empty() else None,
            )
            all_docs.extend(result.documents)
        except Exception as e:
            logger.debug(f"Expanded term search failed for '{term}': {e}")
    
    return all_docs


def _merge_results(
    primary: List[Document],
    secondary: List[Document],
) -> List[Document]:
    """
    Merge primary and secondary results with deduplication.
    """
    seen_ids = {doc.chunk_id for doc in primary}
    merged = list(primary)
    
    for doc in secondary:
        if doc.chunk_id not in seen_ids:
            # Reduce score for secondary results
            doc = doc.model_copy(update={"score": doc.score * 0.8})
            merged.append(doc)
            seen_ids.add(doc.chunk_id)
    
    return merged


# ============================================================================
# Utility Functions
# ============================================================================

def _update_metadata(state: AgentState, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update state metadata with new information.
    """
    metadata = state.get("metadata", {}) or {}
    metadata.update(updates)
    return metadata


# ============================================================================
# LangGraph Workflow Integration
# ============================================================================

def create_retrieval_node():
    """
    Create the retrieval node for LangGraph workflow.
    
    Returns a callable that can be used as a LangGraph node.
    """
    return retrieval_agent


# ============================================================================
# Standalone Execution
# ============================================================================

def retrieve(
    query: str,
    sub_queries: Optional[List[str]] = None,
    target_collections: Optional[List[str]] = None,
    input_source: str = "base_knowledge",
    top_k: int = 10,
) -> List[Document]:
    """
    Standalone retrieval function for direct use.
    
    Args:
        query: Main search query
        sub_queries: Optional decomposed sub-queries
        target_collections: Collections to search (default: all)
        input_source: Source type (base_knowledge, user_text, file_upload)
        top_k: Number of results to return
        
    Returns:
        List of retrieved Document objects
    """
    # Build minimal state
    state: AgentState = {
        "original_query": query,
        "corrected_query": query,
        "normalized_query": query,
        "sub_queries": sub_queries or [query],
        "target_collections": target_collections or ["bukhari", "muslim"],
        "input_source": input_source,
        "query_intent": None,
        "language": None,
        "metadata": {},
    }
    
    # Run retrieval agent
    result = retrieval_agent(state)
    
    return result.get("retrieved_docs", [])[:top_k]


# ============================================================================
# Async API for Advanced Use
# ============================================================================

async def retrieve_async(
    query: str,
    sub_queries: Optional[List[str]] = None,
    target_collections: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Document]:
    """
    Async version of retrieve for use in async contexts.
    """
    sub_queries = sub_queries or [query]
    target_collections = target_collections or ["bukhari", "muslim"]
    
    # Expand queries
    expanded = {}
    expander = QueryExpansionTool()
    for q in sub_queries:
        try:
            result = expander(q, use_llm=False)
            expanded[q] = result.expanded_terms
        except:
            expanded[q] = []
    
    # Execute parallel searches
    all_results = await _execute_parallel_searches(
        sub_queries=sub_queries,
        expanded_terms=expanded,
        target_collections=target_collections,
        top_k=top_k * 2,
    )
    
    # Aggregate
    aggregated = aggregate_results(
        raw_results=all_results,
        original_query=query,
        top_k=top_k,
    )
    
    return aggregated.documents


# ============================================================================
# Testing Entry Point
# ============================================================================

if __name__ == "__main__":
    # Quick test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    test_query = sys.argv[1] if len(sys.argv) > 1 else "أحاديث عن الصلاة"
    
    print(f"\nTesting retrieval for: {test_query}")
    print("=" * 60)
    
    results = retrieve(test_query, top_k=5)
    
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Score: {doc.score:.3f}")
        print(f"   Collection: {doc.collection}")
        print(f"   Hadith ID: {doc.hadith_id}")
        print(f"   Text: {doc.text[:200]}...")
