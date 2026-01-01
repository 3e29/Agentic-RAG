"""
Graph State Definition for Hadith RAG System

This module defines the global AgentState schema used across the LangGraph workflow.
The state is passed between nodes and tracks the query analysis pipeline.

Production Standards:
- Type safety with typing and Literal types
- Enums for fixed categories to avoid magic strings
- Optional fields for conditional execution paths
"""

from enum import Enum
from typing import List, Literal, Optional
from typing_extensions import TypedDict


# ============================================================================
# Enums for Type Safety (Avoiding Magic Strings)
# ============================================================================

class InputSource(str, Enum):
    """
    Source type for the user's input.
    
    Determines how the query should be processed:
    - BASE_KNOWLEDGE: User is querying the hadith database
    - USER_TEXT: User is providing their own text for analysis
    - FILE_UPLOAD: User has uploaded a file for processing
    """
    BASE_KNOWLEDGE = "base_knowledge"
    USER_TEXT = "user_text"
    FILE_UPLOAD = "file_upload"


class CollectionTarget(str, Enum):
    """
    Target hadith collection for search.
    
    Supported collections in the system:
    - BUKHARI: Sahih al-Bukhari collection
    - MUSLIM: Sahih Muslim collection
    """
    BUKHARI = "bukhari"
    MUSLIM = "muslim"


class QueryIntent(str, Enum):
    """
    Classified intent of the user's query.
    
    - THEMATIC_SEARCH: Broad topic search (e.g., "hadiths about prayer")
    - SPECIFIC_LOOKUP: Specific hadith lookup by number, narrator, or text
    - COMPARATIVE_ANALYSIS: Comparing topics, books, or narrators
    - METADATA_QUERY: Queries about hadith metadata (longest, shortest, most narrated, etc.)
    """
    THEMATIC_SEARCH = "thematic_search"
    SPECIFIC_LOOKUP = "specific_lookup"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    METADATA_QUERY = "metadata_query"


class Language(str, Enum):
    """Detected language of the query."""
    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"


# ============================================================================
# Agent State Definition
# ============================================================================

class AgentState(TypedDict):
    """
    Global state for the Hadith RAG LangGraph workflow.
    
    This state is passed between all nodes in the graph and tracks the complete
    query processing pipeline from raw input to retrieval-ready format.
    
    The state supports a conditional pipeline where certain stages may be
    skipped based on intent or input source classification.
    
    Attributes:
        # Core Query Fields
        original_query: The raw user input as received
        normalized_query: Query after Arabic text normalization (no LLM)
        corrected_query: Query after spelling and typo correction (LLM)
        
        # Classification Fields
        input_source: Whether user is querying DB or providing own text
        query_intent: Classified intent (thematic/specific/comparative)
        target_collections: Which hadith collections to search
        
        # Decomposition Fields
        sub_queries: List of decomposed sub-queries for complex questions
        
        # Language & Metadata
        language: Detected language of the query (ar, en, mixed)
        metadata: Additional context and intermediate results for debugging
    """
    
    # -------------------------------------------------------------------------
    # Core Query Fields
    # -------------------------------------------------------------------------
    original_query: str
    normalized_query: Optional[str]  # After regex normalization (no LLM)
    corrected_query: Optional[str]   # After LLM typo correction
    search_query: Optional[str]      # Optimized query for embedding (stripped of question words)
    
    # -------------------------------------------------------------------------
    # Classification Fields  
    # -------------------------------------------------------------------------
    input_source: Optional[Literal["base_knowledge", "user_text", "file_upload"]]
    query_intent: Optional[Literal["thematic_search", "specific_lookup", "comparative_analysis", "metadata_query"]]
    target_collections: Optional[List[Literal["bukhari", "muslim"]]]
    
    # -------------------------------------------------------------------------
    # Decomposition Fields
    # -------------------------------------------------------------------------
    sub_queries: Optional[List[str]]
    
    # -------------------------------------------------------------------------
    # Retrieval Fields
    # -------------------------------------------------------------------------
    retrieved_docs: Optional[List[dict]]  # Documents from retrieval agent
    
    # -------------------------------------------------------------------------
    # Evaluation Fields (Added for Evaluation Agent)
    # -------------------------------------------------------------------------
    evaluation_feedback: Optional[str]              # Actionable feedback from evaluation
    confidence_score: Optional[float]               # Confidence in retrieval quality (0-1)
    missing_information_gaps: Optional[List[str]]   # Identified gaps in results
    
    # -------------------------------------------------------------------------
    # Language & Metadata
    # -------------------------------------------------------------------------
    language: Optional[Literal["ar", "en", "mixed"]]  # Detected query language
    desired_output_language: Literal["arabic", "english"]  # Output language (explicit or inferred from query)
    metadata: Optional[dict]
