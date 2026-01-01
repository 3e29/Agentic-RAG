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
from src.data.hadith_repository import (
    HadithRepository,
    get_hadith_repository,
    reassemble_chunked_hadiths,
)
from src.agents.search_orchestrator import (
    SearchOrchestrator,
    get_search_orchestrator,
)

# Configure logging
logger = logging.getLogger(__name__)

# Import configuration constants
from src.config.settings import (
    MAX_AGENT_ITERATIONS,
    MAX_RETRIES,
    DEFAULT_TOP_K,
    PARALLEL_SEARCH_K,
)

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
    
    Delegates orchestration to SearchOrchestrator while keeping the agent
    decision logic in this module.
    
    The orchestrator handles:
    - Parallel execution of agents for sub-queries
    - Result aggregation and reranking
    - Chunk reassembly
    
    This function provides:
    - State extraction and validation
    - Metadata collection
    - The agent executor function
    """
    # Get sub-queries from state (if query was decomposed)
    sub_queries = state.get("sub_queries") or [query]
    target_collections = state.get("target_collections") or ["bukhari", "muslim"]
    query_language = state.get("language", "ar")
    desired_output_language = state.get("desired_output_language")
    query_intent = state.get("query_intent", "thematic_search")
    
    # Store metadata
    metadata["sub_queries"] = sub_queries
    metadata["target_collections"] = target_collections
    metadata["query_language"] = query_language
    metadata["desired_output_language"] = desired_output_language
    metadata["query_intent"] = query_intent
    metadata["stages"].append("autonomous_search")
    
    logger.info(
        f"Processing {len(sub_queries)} sub-queries via orchestrator "
        f"(lang={query_language}, output={desired_output_language}, intent={query_intent})"
    )
    
    # Get the orchestrator and wire up the agent executor
    orchestrator = get_search_orchestrator()
    orchestrator.set_agent_executor(_execute_autonomous_agent_async)
    
    # Delegate to orchestrator for the full pipeline
    # Handle both sync and async contexts safely
    coro = orchestrator.orchestrate_search(
        sub_queries=sub_queries,
        target_collections=target_collections,
        query_language=query_language,
        desired_output_language=desired_output_language,
        query_intent=query_intent,
        metadata=metadata,
    )
    
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        loop = None
    
    if loop is not None:
        # Already in async context - use nest_asyncio or run in thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            results = future.result()
    else:
        # No async context - use asyncio.run() directly
        results = asyncio.run(coro)
    
    return results


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
    repository: Optional[HadithRepository] = None,
) -> List[Document]:
    """
    Handle metadata-based queries (longest, shortest, most, count, etc.)
    
    These queries require direct database lookups rather than semantic search.
    Uses the HadithRepository for data access.
    
    Supported query types:
    - "longest hadith" -> find hadith with max total_chunks
    - "shortest hadith" -> find hadith with min total_chunks (=1)
    - "how many hadiths" -> count query (future)
    """
    logger.info("Handling metadata query")
    metadata["stages"].append("metadata_query")
    
    target_collections = state.get("target_collections") or ["bukhari", "muslim"]
    desired_language = state.get("desired_output_language")
    
    # 1. Determine query type (Longest vs Shortest)
    query_lower = query.lower()
    is_longest = any(term in query_lower for term in ['أطول', 'اطول', 'longest', 'long'])
    is_shortest = any(term in query_lower for term in ['أقصر', 'اقصر', 'shortest', 'short'])
    
    # If neither longest nor shortest detected, this is NOT a metadata query
    # Fall back to autonomous search (e.g., "ما عدد الصلوات" is asking about content, not hadith stats)
    if not is_longest and not is_shortest:
        logger.info("Metadata query type not detected (not longest/shortest), falling back to autonomous search")
        metadata["stages"].append("metadata_fallback_to_search")
        return _autonomous_search(query, state, metadata)
    
    # 2. Extract Filters & Resolve IDs
    narrator = None
    chapter_id = None
    
    try:
        # Extract raw metadata using LLM
        metadata["stages"].append("extract_filters")
        extracted = extract_metadata_filters(query, use_llm=True)
        narrator = extracted.narrator
        chapter_id = extracted.chapter_id
        
        # --- Resolve Chapter Title to ID if ID is missing ---
        if not chapter_id and (extracted.chapter_title_en or extracted.chapter_title_ar):
            metadata["stages"].append("resolve_chapter_id")
            # Use Arabic title if available (often more precise), otherwise English
            search_title = extracted.chapter_title_ar or extracted.chapter_title_en
            collection_scope = extracted.collection or (target_collections[0] if len(target_collections) == 1 else None)
            
            logger.info(f"Resolving chapter title '{search_title}' to ID (scope: {collection_scope})...")
            
            # Lookup the ID using find_chapter_for_subject
            chapter_result = find_chapter_for_subject(search_title, collection_scope)
            
            if chapter_result:
                chapter_id = chapter_result.chapter_id
                logger.info(f"Resolved '{search_title}' -> chapter_id={chapter_id}")
            else:
                logger.warning(f"Could not resolve chapter title '{search_title}' to an ID")
        
        # --- Narrator Normalization (Arabic -> English) ---
        NARRATOR_MAPPING = {
            "أبو هريرة": "Abu Huraira",
            "ابو هريرة": "Abu Huraira",
            "عائشة": "Aisha",
            "عائشه": "Aisha",
            "أنس": "Anas",
            "أنس بن مالك": "Anas",
            "ابن عمر": "Ibn Umar",
            "عبد الله بن عمر": "Ibn Umar",
            "جابر": "Jabir",
            "أبو سعيد": "Abu Said",
            "ابن عباس": "Ibn Abbas",
            "عبد الله بن عباس": "Ibn Abbas",
            "ابن مسعود": "Ibn Masud",
            "عبد الله بن مسعود": "Ibn Masud",
        }
        
        if narrator and narrator in NARRATOR_MAPPING:
            original_narrator = narrator
            narrator = NARRATOR_MAPPING[narrator]
            logger.info(f"Normalized narrator: {original_narrator} -> {narrator}")
        
        if narrator:
            logger.info(f"Metadata query applying narrator filter: {narrator}")
        if chapter_id:
            logger.info(f"Metadata query applying chapter_id filter: {chapter_id}")
            
    except Exception as e:
        logger.warning(f"Failed to extract/resolve filters for metadata query: {e}")
    
    # 3. Execute Repository Query
    metadata["stages"].append("repository_query")
    repo = repository or get_hadith_repository()
    
    try:
        results = []
        
        for coll_name in target_collections:
            if is_longest:
                doc = repo.get_longest_hadith(
                    collection=coll_name,
                    language=desired_language,
                    narrator=narrator,
                    chapter_id=chapter_id,
                )
            else:
                doc = repo.get_shortest_hadith(
                    collection=coll_name,
                    language=desired_language,
                    narrator=narrator,
                    chapter_id=chapter_id,
                )
            
            if doc:
                results.append(doc)
                
                metadata["metadata_query_result"] = {
                    "query_type": "longest" if is_longest else "shortest",
                    "hadith_id": doc.hadith_id,
                    "total_chunks": doc.total_chunks,
                    "collection": coll_name,
                    "chapter_id": chapter_id,
                }
                
                logger.info(
                    f"Metadata query found: Hadith #{doc.hadith_id} with "
                    f"{doc.total_chunks} chunks ({coll_name})"
                )
        
        return results
        
    except Exception as e:
        logger.error(f"Metadata query failed: {e}")
        metadata["errors"].append({"stage": "metadata_query", "error": str(e)})
        # Fall back to semantic search
        logger.info("Falling back to semantic search")
        return _autonomous_search(query, state, metadata)


# ============================================================================
# Utility Functions
# ============================================================================

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
