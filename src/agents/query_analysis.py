"""
Query Analysis Agent for Hadith RAG System (FR-QAA-04)

This module implements the Query Analysis Agent as a Smart Conditional Pipeline.
It orchestrates the query processing tools in an optimized sequence:

**Pipeline Stages:**
1. Pre-processing (Always, No LLM):
   - Input Source Identification (regex quick-check + LLM fallback)
   - Query Normalization (Arabic text standardization)

2. Correction (Always, LLM):
   - Typo Correction on normalized query

3. Classification (Always, LLM):
   - Intent Classification

4. Targeting (Always, No LLM):
   - Collection Target Detection

5. Decomposition (Conditional, LLM):
   - SKIP if intent == specific_lookup OR input_source != base_knowledge
   - RUN otherwise (thematic_search, comparative_analysis with base_knowledge)

**Production Standards:**
- Type safety with typing
- Defensive programming with try/except and fallback values
- Full observability via logging and LangSmith tracing
- Conditional execution to save tokens and reduce latency
"""

import logging
from typing import Dict, Any, Optional
from langsmith import traceable

from src.graph.state import AgentState, InputSource, QueryIntent
from src.tools.query_processing import (
    query_normalization_tool,
    input_source_identification_tool,
    collection_target_detection_tool,
    typo_correction_tool,
    intent_classification_tool,
    query_decomposition_tool,
)

# Configure logging
logger = logging.getLogger(__name__)


@traceable(name="query_analysis_agent")
def query_analysis_agent(state: AgentState) -> Dict[str, Any]:
    """
    Main Query Analysis Agent node implementing a Smart Conditional Pipeline.
    
    This agent optimizes the query analysis flow by:
    1. Running fast pre-processing steps first (no LLM)
    2. Using conditional logic to skip unnecessary LLM calls
    3. Providing fallback values at every stage
    
    Pipeline:
    1. Pre-processing: InputSourceID + QueryNormalization (fast)
    2. Correction: TypoCorrection (LLM)
    3. Intent: IntentClassification (LLM)
    4. Targeting: CollectionTargetDetection (fast)
    5. Decomposition: QueryDecomposition (CONDITIONAL - LLM)
       - SKIP if intent == specific_lookup
       - SKIP if input_source != base_knowledge
    
    Args:
        state: Current AgentState containing 'original_query'
        
    Returns:
        Dictionary with updated state fields
        
    Raises:
        ValueError: If original_query is missing or empty
    """
    
    # ========================================================================
    # Input Validation
    # ========================================================================
    
    original_query = state.get("original_query")
    if not original_query or not original_query.strip():
        logger.error("Query analysis called with empty query")
        raise ValueError("original_query is required and cannot be empty")
    
    logger.info(f"Starting query analysis for: {original_query[:100]}...")
    
    # Initialize metadata tracking
    metadata = state.get("metadata", {}) or {}
    metadata["query_analysis"] = {
        "stages_completed": [],
        "stages_skipped": [],
        "errors": [],
        "pipeline_version": "2.0-conditional"
    }
    
    # ========================================================================
    # Stage 1: Pre-processing (Fast Path - No LLM for Normalization)
    # ========================================================================
    
    logger.info("=== Stage 1/5: Pre-processing ===")
    
    # 1a. Input Source Identification
    logger.info("Stage 1a: Input Source Identification")
    try:
        source_result = input_source_identification_tool(original_query)
        input_source = source_result.source_type
        
        metadata["query_analysis"]["input_source"] = {
            "source_type": input_source,
            "confidence": source_result.confidence,
            "reasoning": source_result.reasoning
        }
        metadata["query_analysis"]["stages_completed"].append("input_source_identification")
        
        logger.info(f"Input source identified: {input_source} (confidence: {source_result.confidence:.2f})")
        
    except Exception as e:
        logger.error(f"Input source identification failed: {e}")
        metadata["query_analysis"]["errors"].append({
            "stage": "input_source_identification",
            "error": str(e)
        })
        input_source = "base_knowledge"  # Default fallback
        logger.warning("Using default input source: base_knowledge")
    
    # 1b. Query Normalization (Pure regex - very fast)
    logger.info("Stage 1b: Query Normalization")
    try:
        norm_result = query_normalization_tool(original_query)
        normalized_query = norm_result.normalized_text
        
        metadata["query_analysis"]["normalization"] = {
            "original": original_query,
            "normalized": normalized_query,
            "transformations": norm_result.transformations_applied
        }
        metadata["query_analysis"]["stages_completed"].append("query_normalization")
        
        logger.info(f"Query normalized. {len(norm_result.transformations_applied)} transformation(s) applied.")
        
    except Exception as e:
        logger.error(f"Query normalization failed: {e}")
        metadata["query_analysis"]["errors"].append({
            "stage": "query_normalization",
            "error": str(e)
        })
        normalized_query = original_query  # Fallback
        logger.warning("Using original query after normalization failure")
    
    # ========================================================================
    # Stage 2: Typo Correction (LLM)
    # ========================================================================
    
    logger.info("=== Stage 2/5: Typo Correction ===")
    
    # Initialize desired_output_language
    desired_output_language = None
    
    try:
        typo_result = typo_correction_tool(normalized_query)
        corrected_query = typo_result.corrected_text
        language = typo_result.language
        desired_output_language = typo_result.desired_output_language
        
        metadata["query_analysis"]["typo_correction"] = {
            "input": normalized_query,
            "corrected": corrected_query,
            "language": language,
            "desired_output_language": desired_output_language,
            "corrections_made": typo_result.corrections_made
        }
        metadata["query_analysis"]["stages_completed"].append("typo_correction")
        
        logger.info(f"Typo correction complete. Language: {language}, Desired output: {desired_output_language}")
        
    except Exception as e:
        logger.error(f"Typo correction failed: {e}")
        metadata["query_analysis"]["errors"].append({
            "stage": "typo_correction",
            "error": str(e)
        })
        corrected_query = normalized_query
        language = "en"
        desired_output_language = None
        logger.warning("Using normalized query after typo correction failure")
    
    # ========================================================================
    # Stage 3: Intent Classification (LLM)
    # ========================================================================
    
    logger.info("=== Stage 3/5: Intent Classification ===")
    
    try:
        intent_result = intent_classification_tool(corrected_query)
        query_intent = intent_result.intent
        
        metadata["query_analysis"]["intent_classification"] = {
            "intent": query_intent,
            "confidence": intent_result.confidence,
            "reasoning": intent_result.reasoning
        }
        metadata["query_analysis"]["stages_completed"].append("intent_classification")
        
        logger.info(f"Intent classified: {query_intent} (confidence: {intent_result.confidence:.2f})")
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        metadata["query_analysis"]["errors"].append({
            "stage": "intent_classification",
            "error": str(e)
        })
        query_intent = "thematic_search"
        logger.warning("Using default intent: thematic_search")
    
    # ========================================================================
    # Stage 4: Collection Target Detection (Fast - No LLM)
    # ========================================================================
    
    logger.info("=== Stage 4/5: Collection Target Detection ===")
    
    try:
        target_result = collection_target_detection_tool(corrected_query)
        target_collections = target_result.targets
        
        metadata["query_analysis"]["collection_targeting"] = {
            "targets": target_collections,
            "reasoning": target_result.reasoning
        }
        metadata["query_analysis"]["stages_completed"].append("collection_target_detection")
        
        logger.info(f"Target collections: {target_collections}")
        
    except Exception as e:
        logger.error(f"Collection target detection failed: {e}")
        metadata["query_analysis"]["errors"].append({
            "stage": "collection_target_detection",
            "error": str(e)
        })
        target_collections = ["bukhari", "muslim"]
        logger.warning("Using all collections as fallback")
    
    # ========================================================================
    # Stage 5: Query Decomposition (CONDITIONAL - LLM)
    # ========================================================================
    
    logger.info("=== Stage 5/5: Query Decomposition (Conditional) ===")
    
    # Determine if decomposition should be skipped
    skip_decomposition = False
    skip_reason = None
    
    # Condition 1: Skip for specific_lookup intent
    if query_intent == "specific_lookup":
        skip_decomposition = True
        skip_reason = "specific_lookup intent - targeted query doesn't benefit from decomposition"
    
    # Condition 2: Skip if input_source is not base_knowledge
    elif input_source != "base_knowledge":
        skip_decomposition = True
        skip_reason = f"input_source is '{input_source}' - not a database query"
    
    if skip_decomposition:
        logger.info(f"Decomposition SKIPPED: {skip_reason}")
        
        sub_queries = None
        
        metadata["query_analysis"]["query_decomposition"] = {
            "skipped": True,
            "skip_reason": skip_reason,
            "is_complex": False,
            "sub_queries_count": 0
        }
        metadata["query_analysis"]["stages_skipped"].append("query_decomposition")
        
    else:
        # Run decomposition for thematic_search and comparative_analysis
        logger.info(f"Running decomposition (intent: {query_intent}, source: {input_source})")
        
        try:
            decomp_result = query_decomposition_tool(corrected_query)
            
            if decomp_result.is_complex:
                sub_queries = decomp_result.sub_queries
                logger.info(f"Complex query decomposed into {len(sub_queries)} sub-queries")
            else:
                sub_queries = None
                logger.info("Query is simple, no decomposition needed")
            
            metadata["query_analysis"]["query_decomposition"] = {
                "skipped": False,
                "is_complex": decomp_result.is_complex,
                "sub_queries_count": len(decomp_result.sub_queries) if decomp_result.sub_queries else 0,
                "reasoning": decomp_result.reasoning
            }
            metadata["query_analysis"]["stages_completed"].append("query_decomposition")
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            metadata["query_analysis"]["errors"].append({
                "stage": "query_decomposition",
                "error": str(e)
            })
            sub_queries = None
            logger.warning("Treating query as simple after decomposition failure")
    
    # ========================================================================
    # Finalize and Return Updated State
    # ========================================================================
    
    stages_run = len(metadata["query_analysis"]["stages_completed"])
    stages_skipped = len(metadata["query_analysis"]["stages_skipped"])
    errors_count = len(metadata["query_analysis"]["errors"])
    
    logger.info(f"Query analysis complete. {stages_run} stages run, {stages_skipped} skipped, {errors_count} errors.")
    
    # Build the state update
    state_update = {
        "original_query": original_query,
        "normalized_query": normalized_query,
        "corrected_query": corrected_query,
        "input_source": input_source,
        "query_intent": query_intent,
        "target_collections": target_collections,
        "sub_queries": sub_queries,
        "language": language,
        "desired_output_language": desired_output_language,  # User's preferred result language
        "metadata": metadata
    }
    
    # Log summary
    decomp_status = f"skipped ({skip_reason[:30]}...)" if skip_decomposition else f"{len(sub_queries) if sub_queries else 0} sub-queries"
    logger.info(
        f"\n{'='*60}\n"
        f"ANALYSIS SUMMARY\n"
        f"{'='*60}\n"
        f"  Original:     {original_query[:60]}...\n"
        f"  Normalized:   {normalized_query[:60]}...\n"
        f"  Corrected:    {corrected_query[:60]}...\n"
        f"  Language:     {language}\n"
        f"  Desired Output: {desired_output_language}\n"
        f"  Input Source: {input_source}\n"
        f"  Intent:       {query_intent}\n"
        f"  Collections:  {target_collections}\n"
        f"  Decomposition: {decomp_status}\n"
        f"  Errors:       {errors_count}\n"
        f"{'='*60}"
    )
    
    return state_update


# ============================================================================
# Graph Building Utility
# ============================================================================

def create_query_analysis_workflow():
    """
    Create a minimal LangGraph workflow with just the query analysis node.
    
    Useful for testing the agent in isolation before integrating
    with the full RAG pipeline.
    
    Returns:
        Compiled LangGraph with query_analysis_agent as the main node
    """
    from langgraph.graph import StateGraph, START, END
    from src.graph.state import AgentState
    
    builder = StateGraph(AgentState)
    builder.add_node("query_analysis", query_analysis_agent)
    builder.add_edge(START, "query_analysis")
    builder.add_edge("query_analysis", END)
    
    graph = builder.compile()
    
    logger.info("Query analysis workflow created")
    return graph


# ============================================================================
# Convenience Function for Direct Invocation
# ============================================================================

def analyze_query(query: str) -> Dict[str, Any]:
    """
    Convenience function to analyze a query without using LangGraph.
    
    Useful for quick testing or embedding the agent in non-graph contexts.
    
    Args:
        query: Raw user query string
        
    Returns:
        Dictionary containing analysis results
    """
    
    initial_state: AgentState = {
        "original_query": query,
        "normalized_query": None,
        "corrected_query": None,
        "input_source": None,
        "query_intent": None,
        "target_collections": None,
        "sub_queries": None,
        "language": None,
        "metadata": {}
    }
    
    result = query_analysis_agent(initial_state)
    final_state = {**initial_state, **result}
    
    return final_state
