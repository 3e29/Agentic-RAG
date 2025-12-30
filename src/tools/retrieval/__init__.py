"""
Retrieval Tools Package for Hadith RAG System

This package implements the 7 core retrieval tools following Clean Architecture:

**Search Tools:**
1. SemanticSearchTool (FR-RA-12) - Vector similarity search
2. KeywordSearchTool (FR-RA-13) - BM25 lexical search  
3. HybridSearchTool (FR-RA-15) - Combined with RRF fusion

**Filter & Processing Tools:**
4. MetadataFilterTool (FR-RA-14) - Convert NL constraints to DB filters
5. QueryExpansionTool - Generate synonyms/translations

**Aggregation Tools:**
6. ResultAggregationTool - Deduplicate and rerank results

**User Content Tools:**
7. UserHadithProcessorTool (FR-RA-17) - Process user-provided text

Production Standards:
- Pydantic V2 for all schemas
- Dependency injection for vector store/BM25
- Graceful error handling with fallbacks
- Full observability via logging and tracing
"""

from src.tools.retrieval.search_tools import (
    SemanticSearchTool,
    KeywordSearchTool,
    HybridSearchTool,
    semantic_search,
    keyword_search,
    hybrid_search,
)
from src.tools.retrieval.filter_tools import (
    MetadataFilterTool,
    QueryExpansionTool,
    extract_metadata_filters,
    expand_query,
)
from src.tools.retrieval.aggregation_tools import (
    ResultAggregationTool,
    aggregate_results,
)
from src.tools.retrieval.user_content_tools import (
    UserHadithProcessorTool,
    process_user_hadith,
)
from src.tools.retrieval.schemas import (
    Document,
    SearchResult,
    MetadataFilter,
    AggregatedResults,
)

__all__ = [
    # Search Tools
    "SemanticSearchTool",
    "KeywordSearchTool", 
    "HybridSearchTool",
    "semantic_search",
    "keyword_search",
    "hybrid_search",
    # Filter Tools
    "MetadataFilterTool",
    "QueryExpansionTool",
    "extract_metadata_filters",
    "expand_query",
    # Aggregation Tools
    "ResultAggregationTool",
    "aggregate_results",
    # User Content Tools
    "UserHadithProcessorTool",
    "process_user_hadith",
    # Schemas
    "Document",
    "SearchResult",
    "MetadataFilter",
    "AggregatedResults",
]
