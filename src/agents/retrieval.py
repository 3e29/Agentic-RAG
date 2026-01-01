"""
Retrieval Agent for Hadith RAG System (Refactored v4.0)

This module implements the Retrieval Agent as a **single-pass search executor**.
The ReAct loop logic has been moved to the LangGraph workflow, where the
Evaluation Agent decides whether to retry.

**Architecture (Separation of Concerns):**
- Retrieval Agent: "Execute this search strategy, return docs"
- Evaluation Agent: "Assess quality, decide CONTINUE/STOP"
- Workflow (LangGraph): "Route between agents based on decisions"

**Key Changes from v3.x:**
- REMOVED: Internal ReAct loop and `_get_agent_decision`
- SIMPLIFIED: Agent executes ONE search pass per invocation
- ADDED: Support for feedback-driven retry strategies

**Supported Search Strategies:**
1. default: Hybrid search (semantic + keyword)
2. expand_query: Expand query terms first, then hybrid search
3. keyword_search: BM25 keyword-only search
4. semantic_search: Vector-only search
5. relax_filters: Search with relaxed metadata filters

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
# Retrieval Agent (Single-Pass Executor)
# ============================================================================

@traceable(name="retrieval_agent", run_type="chain")
def retrieval_agent(state: AgentState) -> Dict[str, Any]:
    """
    Retrieval Agent node for LangGraph workflow.
    
    Executes a SINGLE search pass based on the current state and any
    feedback from the Evaluation Agent. The workflow (not this agent)
    controls whether to retry.
    
    **Query Priority:**
    - search_query: Optimized for embedding (stripped of question words)
    - corrected_query: Full corrected query (fallback)
    - normalized_query / original_query: Raw inputs (last resort)
    
    **Routing Logic:**
    1. user_text/file_upload -> UserHadithProcessor
    2. metadata_query (longest/shortest) -> HadithRepository
    3. Otherwise -> Hybrid/Semantic/Keyword search based on feedback
    
    Args:
        state: Current AgentState with query analysis results
        
    Returns:
        Dictionary with 'retrieved_docs' and updated 'metadata'
    """
    start_time = time.time()
    
    # Get the OPTIMIZED search query for embedding (priority order)
    search_query = (
        state.get("search_query") or  # Best: optimized for embedding
        state.get("corrected_query") or 
        state.get("normalized_query") or 
        state.get("original_query", "")
    )
    
    # Also keep original/corrected query for logging and context
    original_query = state.get("original_query", "")
    corrected_query = state.get("corrected_query", search_query)
    
    if not search_query:
        logger.error("Retrieval agent called with no query")
        return {
            "retrieved_docs": [],
            "metadata": _update_metadata(state, {"error": "No query provided"})
        }
    
    logger.info(f"Starting retrieval with search_query: '{search_query[:100]}...'")
    if search_query != corrected_query:
        logger.info(f"(Original corrected query was: '{corrected_query[:100]}...')")
    
    # Initialize metadata
    retrieval_metadata = {
        "agent": "retrieval_v4",
        "search_query_used": search_query,  # The optimized query for embedding
        "original_query": original_query,    # For reference
        "stages": [],
        "errors": [],
        "search_strategy": "default",
    }
    
    # Check for evaluation feedback (if this is a retry)
    evaluation_feedback = state.get("evaluation_feedback")
    suggested_actions = []
    if state.get("metadata", {}).get("evaluation", {}).get("suggested_actions"):
        suggested_actions = state["metadata"]["evaluation"]["suggested_actions"]
    
    if evaluation_feedback:
        retrieval_metadata["retry_reason"] = evaluation_feedback
        retrieval_metadata["suggested_actions"] = suggested_actions
        logger.info(f"Retry requested with feedback: {evaluation_feedback}")
    
    # Route based on input source and intent
    input_source = state.get("input_source", "base_knowledge")
    query_intent = state.get("query_intent", "thematic_search")
    
    if input_source == "user_text":
        result = _handle_user_text(search_query, state, retrieval_metadata)
    elif input_source == "file_upload":
        result = _handle_user_text(search_query, state, retrieval_metadata)
    elif query_intent == "metadata_query":
        # For metadata queries, use corrected_query (may need collection names)
        result = _handle_metadata_query(corrected_query, state, retrieval_metadata)
    else:
        # Execute search based on feedback or default strategy
        strategy = _determine_search_strategy(suggested_actions, evaluation_feedback)
        retrieval_metadata["search_strategy"] = strategy
        result = _execute_search_strategy(search_query, state, retrieval_metadata, strategy)
    
    execution_time = (time.time() - start_time) * 1000
    retrieval_metadata["total_execution_time_ms"] = execution_time
    
    logger.info(
        f"Retrieval complete: {len(result)} documents in {execution_time:.1f}ms"
    )
    
    return {
        "retrieved_docs": result,
        "metadata": _update_metadata(state, {"retrieval": retrieval_metadata})
    }


# ============================================================================
# Search Strategy Execution
# ============================================================================

def _determine_search_strategy(
    suggested_actions: List[str],
    feedback: Optional[str],
) -> str:
    """
    Determine the search strategy based on evaluation feedback.
    
    Maps suggested actions from Evaluation Agent to search strategies.
    
    Args:
        suggested_actions: List of actions suggested by Evaluation Agent
        feedback: Human-readable feedback string
        
    Returns:
        Search strategy name
    """
    if not suggested_actions:
        return "default"
    
    # Priority order for actions
    action_to_strategy = {
        "expand_query": "expand_query",
        "relax_filters": "relax_filters",
        "keyword_search": "keyword_search",
        "semantic_search": "semantic_search",
        "hybrid_search": "default",
        "find_chapter": "find_chapter",
    }
    
    for action in suggested_actions:
        if action in action_to_strategy:
            return action_to_strategy[action]
    
    return "default"


@traceable(name="execute_search_strategy", run_type="chain")
def _execute_search_strategy(
    query: str,
    state: AgentState,
    metadata: Dict[str, Any],
    strategy: str,
) -> List[Document]:
    """
    Execute the specified search strategy.
    
    This is the core search execution function that replaces the
    old ReAct loop. Each strategy runs ONCE per invocation.
    
    Args:
        query: The search query
        state: Current agent state
        metadata: Metadata dict to update
        strategy: Strategy to execute
        
    Returns:
        List of retrieved documents
    """
    target_collections = state.get("target_collections") or ["bukhari", "muslim"]
    desired_output_language = state.get("desired_output_language")
    query_intent = state.get("query_intent", "thematic_search")
    sub_queries = state.get("sub_queries") or [query]
    
    metadata["stages"].append(f"strategy_{strategy}")
    metadata["target_collections"] = target_collections
    
    # Build initial filters
    filters = MetadataFilter()
    if len(target_collections) == 1:
        filters.collection = target_collections[0]
    if desired_output_language:
        filters.language = desired_output_language
    
    # Execute based on strategy
    if strategy == "expand_query":
        return _strategy_expand_and_search(query, sub_queries, filters, metadata)
    
    elif strategy == "keyword_search":
        return _strategy_keyword_search(query, sub_queries, filters, metadata)
    
    elif strategy == "semantic_search":
        return _strategy_semantic_search(query, sub_queries, filters, metadata)
    
    elif strategy == "relax_filters":
        return _strategy_relax_and_search(query, sub_queries, filters, metadata)
    
    elif strategy == "find_chapter":
        return _strategy_find_chapter_and_search(query, sub_queries, filters, target_collections, metadata)
    
    else:  # default - hybrid search
        return _strategy_hybrid_search(query, sub_queries, filters, metadata)


@traceable(name="strategy_hybrid_search", run_type="chain")
def _strategy_hybrid_search(
    query: str,
    sub_queries: List[str],
    filters: MetadataFilter,
    metadata: Dict[str, Any],
) -> List[Document]:
    """Default hybrid search strategy."""
    metadata["stages"].append("hybrid_search")
    
    all_results = []
    for sub_query in sub_queries:
        results = _agent_hybrid_search(sub_query, filters, PARALLEL_SEARCH_K)
        all_results.extend(results)
    
    # Aggregate and deduplicate
    aggregated = aggregate_results(
        raw_results=[all_results],
        original_query=query,
        top_k=DEFAULT_TOP_K,
        use_reranker=True,
    )
    
    return aggregated.documents


@traceable(name="strategy_expand_and_search", run_type="chain")
def _strategy_expand_and_search(
    query: str,
    sub_queries: List[str],
    filters: MetadataFilter,
    metadata: Dict[str, Any],
) -> List[Document]:
    """Expand query terms, then hybrid search."""
    metadata["stages"].append("expand_query")
    
    # Expand the main query
    expanded_terms, expansion_result = _agent_expand_query(query)
    metadata["expansion"] = {"terms": expanded_terms, "result": expansion_result}
    
    # Build expanded query
    if expanded_terms:
        expanded_query = ' '.join([query] + expanded_terms[:3])
    else:
        expanded_query = query
    
    # Execute hybrid search with expanded query
    metadata["stages"].append("hybrid_search_expanded")
    all_results = []
    
    for sub_query in sub_queries:
        # Use expanded version of main query
        search_query = expanded_query if sub_query == query else sub_query
        results = _agent_hybrid_search(search_query, filters, PARALLEL_SEARCH_K)
        all_results.extend(results)
    
    aggregated = aggregate_results(
        raw_results=[all_results],
        original_query=query,
        top_k=DEFAULT_TOP_K,
        use_reranker=True,
    )
    
    return aggregated.documents


@traceable(name="strategy_keyword_search", run_type="chain")
def _strategy_keyword_search(
    query: str,
    sub_queries: List[str],
    filters: MetadataFilter,
    metadata: Dict[str, Any],
) -> List[Document]:
    """Keyword-only (BM25) search strategy."""
    metadata["stages"].append("keyword_search")
    
    all_results = []
    for sub_query in sub_queries:
        results = _agent_keyword_search(sub_query, filters, PARALLEL_SEARCH_K)
        all_results.extend(results)
    
    aggregated = aggregate_results(
        raw_results=[all_results],
        original_query=query,
        top_k=DEFAULT_TOP_K,
        use_reranker=True,
    )
    
    return aggregated.documents


@traceable(name="strategy_semantic_search", run_type="chain")
def _strategy_semantic_search(
    query: str,
    sub_queries: List[str],
    filters: MetadataFilter,
    metadata: Dict[str, Any],
) -> List[Document]:
    """Semantic-only (vector) search strategy."""
    metadata["stages"].append("semantic_search")
    
    all_results = []
    for sub_query in sub_queries:
        results = _agent_semantic_search(sub_query, filters, PARALLEL_SEARCH_K)
        all_results.extend(results)
    
    aggregated = aggregate_results(
        raw_results=[all_results],
        original_query=query,
        top_k=DEFAULT_TOP_K,
        use_reranker=True,
    )
    
    return aggregated.documents


@traceable(name="strategy_relax_and_search", run_type="chain")
def _strategy_relax_and_search(
    query: str,
    sub_queries: List[str],
    filters: MetadataFilter,
    metadata: Dict[str, Any],
) -> List[Document]:
    """Relax filters, then hybrid search."""
    metadata["stages"].append("relax_filters")
    
    # Relax the filters
    relaxed_filters = filters.relax(level=1) if filters else None
    metadata["filters_relaxed"] = True
    
    # Execute hybrid search with relaxed filters
    metadata["stages"].append("hybrid_search_relaxed")
    all_results = []
    
    for sub_query in sub_queries:
        results = _agent_hybrid_search(sub_query, relaxed_filters, PARALLEL_SEARCH_K)
        all_results.extend(results)
    
    aggregated = aggregate_results(
        raw_results=[all_results],
        original_query=query,
        top_k=DEFAULT_TOP_K,
        use_reranker=True,
    )
    
    return aggregated.documents


@traceable(name="strategy_find_chapter_and_search", run_type="chain")
def _strategy_find_chapter_and_search(
    query: str,
    sub_queries: List[str],
    filters: MetadataFilter,
    target_collections: List[str],
    metadata: Dict[str, Any],
) -> List[Document]:
    """Find relevant chapter first, then search within it."""
    metadata["stages"].append("find_chapter")
    
    # Find chapter
    collection = target_collections[0] if len(target_collections) == 1 else None
    chapter_id, chapter_result = _agent_find_chapter(query, collection)
    metadata["chapter_lookup"] = {"chapter_id": chapter_id, "result": chapter_result}
    
    # Update filters with chapter
    if chapter_id:
        filters.chapter_id = chapter_id
    
    # Execute hybrid search with chapter filter
    metadata["stages"].append("hybrid_search_chapter")
    all_results = []
    
    for sub_query in sub_queries:
        results = _agent_hybrid_search(sub_query, filters, PARALLEL_SEARCH_K)
        all_results.extend(results)
    
    aggregated = aggregate_results(
        raw_results=[all_results],
        original_query=query,
        top_k=DEFAULT_TOP_K,
        use_reranker=True,
    )
    
    return aggregated.documents


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
    # Fall back to default search (e.g., "ما عدد الصلوات" is asking about content, not hadith stats)
    if not is_longest and not is_shortest:
        logger.info("Metadata query type not detected (not longest/shortest), falling back to default search")
        metadata["stages"].append("metadata_fallback_to_search")
        return _execute_search_strategy(query, state, metadata, "default")
    
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
        # Fall back to default search
        logger.info("Falling back to default search strategy")
        return _execute_search_strategy(query, state, metadata, "default")


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
