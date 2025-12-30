"""
Pydantic V2 Schemas for Retrieval Tools

This module defines all data contracts for the retrieval system using Pydantic V2.
These schemas ensure type safety and validation across all retrieval operations.

Production Standards:
- Strict validation with Pydantic V2
- Field descriptions for documentation
- Default values for optional fields
- Serialization-ready for API responses
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================================
# Enums for Type Safety
# ============================================================================

class SearchType(str, Enum):
    """Type of search performed."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class CollectionName(str, Enum):
    """Supported hadith collections."""
    BUKHARI = "bukhari"
    MUSLIM = "muslim"
    ALL = "all"


# ============================================================================
# Core Document Schema
# ============================================================================

class Document(BaseModel):
    """
    Represents a retrieved hadith document.
    
    This is the core schema returned by all search operations.
    Contains the hadith text, metadata, and relevance scores.
    """
    model_config = ConfigDict(extra="allow")
    
    # Core identifiers
    chunk_id: str = Field(description="Unique identifier for the document chunk")
    hadith_id: Optional[int] = Field(default=None, description="Original hadith ID in collection")
    parent_hadith_id: Optional[str] = Field(default=None, description="Parent hadith reference")
    
    # Content
    text: str = Field(description="The hadith text content")
    language: str = Field(default="arabic", description="Language of the text (arabic/english)")
    
    # Metadata
    collection: str = Field(default="", description="Source collection (Bukhari/Muslim)")
    book_id: Optional[int] = Field(default=None, description="Book ID within collection")
    book_number: Optional[str] = Field(default=None, description="Book number")
    chapter_id: Optional[int] = Field(default=None, description="Chapter ID within book")
    chapter_number: Optional[str] = Field(default=None, description="Chapter number")
    narrator: Optional[str] = Field(default=None, description="Chain of narration")
    hadith_id_in_book: Optional[int] = Field(default=None, description="Hadith number within book")
    
    # Chunking info
    chunk_index: int = Field(default=0, description="Index of this chunk")
    total_chunks: int = Field(default=1, description="Total chunks for parent hadith")
    is_chunked: bool = Field(default=False, description="Whether hadith was split into chunks")
    
    # Search relevance
    score: float = Field(default=0.0, description="Relevance score from search")
    search_type: Optional[str] = Field(default=None, description="Type of search that found this")
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Document text cannot be empty")
        return v
    
    @field_validator('score')
    @classmethod
    def score_valid_range(cls, v: float) -> float:
        # Allow scores outside 0-1 for BM25 and RRF
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()


# ============================================================================
# Search Result Schemas
# ============================================================================

class SearchResult(BaseModel):
    """
    Result from a single search operation.
    
    Contains the documents found and metadata about the search.
    """
    documents: List[Document] = Field(default_factory=list, description="Retrieved documents")
    query: str = Field(description="The query that was searched")
    search_type: SearchType = Field(description="Type of search performed")
    total_found: int = Field(default=0, description="Total documents found before filtering")
    filters_applied: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters used")
    execution_time_ms: Optional[float] = Field(default=None, description="Search execution time")
    
    @field_validator('documents', mode='before')
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v


class HybridSearchResult(BaseModel):
    """
    Result from hybrid search combining semantic and keyword results.
    
    Includes fusion scores and component results for transparency.
    """
    documents: List[Document] = Field(default_factory=list, description="Fused ranked documents")
    query: str = Field(description="The query searched")
    semantic_results: List[Document] = Field(default_factory=list, description="Raw semantic results")
    keyword_results: List[Document] = Field(default_factory=list, description="Raw keyword results")
    alpha: float = Field(default=0.5, description="Semantic weight in fusion (0-1)")
    total_found: int = Field(default=0, description="Total unique documents")
    filters_applied: Optional[Dict[str, Any]] = Field(default=None)
    execution_time_ms: Optional[float] = Field(default=None)


# ============================================================================
# Metadata Filter Schema
# ============================================================================

class MetadataFilter(BaseModel):
    """
    Structured metadata filters for database queries.
    
    Converted from natural language constraints by MetadataFilterTool.
    
    Note on hadith IDs:
    - hadith_id: Internal unique ID across the entire collection
    - hadith_id_in_book: The hadith number within a specific book (what users typically reference)
    When users ask for "hadith number 70", they mean hadith_id_in_book, not hadith_id.
    """
    collection: Optional[str] = Field(default=None, description="Target collection (bukhari/muslim)")
    book_id: Optional[int] = Field(default=None, description="Specific book ID")
    book_number: Optional[str] = Field(default=None, description="Book number string")
    chapter_id: Optional[int] = Field(default=None, description="Specific chapter ID")
    chapter_number: Optional[str] = Field(default=None, description="Chapter number string")
    chapter_title_en: Optional[str] = Field(default=None, description="English chapter title")
    chapter_title_ar: Optional[str] = Field(default=None, description="Arabic chapter title")
    narrator: Optional[str] = Field(default=None, description="Narrator name to filter by")
    hadith_id: Optional[int] = Field(default=None, description="Internal hadith ID (use hadith_id_in_book for user queries)")
    hadith_id_in_book: Optional[int] = Field(default=None, description="Hadith number within the book - what users typically reference")
    language: Optional[str] = Field(default=None, description="Language filter (arabic/english)")
    
    confidence: float = Field(default=1.0, description="Confidence in filter extraction")
    raw_constraints: Optional[str] = Field(default=None, description="Original NL constraints")
    
    def to_chroma_filter(self) -> Optional[Dict[str, Any]]:
        """
        Convert to ChromaDB where filter format.
        
        Returns None if no filters are set.
        """
        conditions = []
        
        if self.collection:
            # Normalize collection name
            coll = self.collection.lower()
            if "bukhari" in coll:
                conditions.append({"collection": {"$eq": "Sahih al-Bukhari"}})
            elif "muslim" in coll:
                conditions.append({"collection": {"$eq": "Sahih Muslim"}})
        
        if self.book_id is not None:
            conditions.append({"book_id": {"$eq": self.book_id}})
            
        if self.chapter_id is not None:
            conditions.append({"chapter_id": {"$eq": self.chapter_id}})
            
        if self.chapter_title_en:
            conditions.append({"chapter_title_en": {"$contains": self.chapter_title_en}})
            
        if self.chapter_title_ar:
            conditions.append({"chapter_title_ar": {"$contains": self.chapter_title_ar}})
            
        if self.narrator:
            conditions.append({"narrator": {"$contains": self.narrator}})
        
        # Prefer hadith_id_in_book (user-facing number) over internal hadith_id
        if self.hadith_id_in_book is not None:
            conditions.append({"hadith_id_in_book": {"$eq": self.hadith_id_in_book}})
            
        if self.language:
            conditions.append({"language": {"$eq": self.language}})
        
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    def relax(self, level: int = 1) -> "MetadataFilter":
        """
        Create a relaxed version of filters for retry.
        
        Level 1: Remove chapter filter (ID and Title)
        Level 2: Remove book filter
        Level 3: Remove all except collection
        """
        relaxed = self.model_copy()
        
        if level >= 1:
            relaxed.chapter_id = None
            relaxed.chapter_number = None
            relaxed.chapter_title_en = None
            relaxed.chapter_title_ar = None
            
        if level >= 2:
            relaxed.book_id = None
            relaxed.book_number = None
            
        if level >= 3:
            relaxed.narrator = None
            relaxed.hadith_id = None
            relaxed.language = None
            
        return relaxed
    
    def is_empty(self) -> bool:
        """Check if no meaningful filters are set."""
        return all([
            self.collection is None,
            self.book_id is None,
            self.chapter_id is None,
            self.chapter_title_en is None,
            self.chapter_title_ar is None,
            self.narrator is None,
            self.hadith_id is None,
        ])


# ============================================================================
# Aggregation Result Schema
# ============================================================================

class AggregatedResults(BaseModel):
    """
    Final aggregated and reranked results from multiple sub-queries.
    
    This is the output of the ResultAggregationTool after deduplication
    and cross-encoder reranking.
    """
    documents: List[Document] = Field(default_factory=list, description="Final ranked documents")
    total_unique: int = Field(default=0, description="Number of unique documents")
    duplicates_removed: int = Field(default=0, description="Number of duplicates merged")
    sub_query_counts: Dict[str, int] = Field(
        default_factory=dict, 
        description="Documents contributed by each sub-query"
    )
    reranking_applied: bool = Field(default=False, description="Whether cross-encoder reranking was used")
    execution_time_ms: Optional[float] = Field(default=None)
    
    def get_top_k(self, k: int = 10) -> List[Document]:
        """Get top K documents from results."""
        return self.documents[:k]


# ============================================================================
# Query Expansion Schema
# ============================================================================

class ExpandedQuery(BaseModel):
    """
    Result of query expansion with synonyms and translations.
    """
    original_query: str = Field(description="Original input query")
    expanded_terms: List[str] = Field(default_factory=list, description="Additional search terms")
    translations: Dict[str, str] = Field(
        default_factory=dict, 
        description="Translations (e.g., Arabic to English)"
    )
    confidence: float = Field(default=1.0, description="Confidence in expansion quality")
    
    def get_all_queries(self) -> List[str]:
        """Get all query variations for searching."""
        queries = [self.original_query]
        queries.extend(self.expanded_terms)
        queries.extend(self.translations.values())
        return list(set(queries))  # Deduplicate


# ============================================================================
# User Content Processing Schema
# ============================================================================

class UserContentResult(BaseModel):
    """
    Result of processing user-provided hadith text.
    """
    original_text: str = Field(description="User's original text input")
    processed_text: str = Field(description="Cleaned/normalized text")
    similar_hadiths: List[Document] = Field(
        default_factory=list, 
        description="Similar hadiths found in knowledge base"
    )
    is_authentic: Optional[bool] = Field(
        default=None, 
        description="Whether text matches authentic hadith"
    )
    match_confidence: float = Field(default=0.0, description="Confidence in authenticity match")
    indexed: bool = Field(default=False, description="Whether text was indexed temporarily")


# ============================================================================
# Tool Input Schemas
# ============================================================================

class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""
    query: str = Field(description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: Optional[MetadataFilter] = Field(default=None, description="Optional metadata filters")


class KeywordSearchInput(BaseModel):
    """Input schema for keyword/BM25 search tool."""
    query: str = Field(description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: Optional[MetadataFilter] = Field(default=None, description="Optional metadata filters")


class HybridSearchInput(BaseModel):
    """Input schema for hybrid search tool."""
    query: str = Field(description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Semantic weight (0=keyword only, 1=semantic only)")
    filters: Optional[MetadataFilter] = Field(default=None, description="Optional metadata filters")


class AggregationInput(BaseModel):
    """Input schema for result aggregation tool."""
    raw_results: List[List[Document]] = Field(description="Results from multiple sub-queries")
    original_query: str = Field(description="Original user query for reranking context")
    top_k: int = Field(default=20, ge=1, le=100, description="Final number of results to return")
    use_reranker: bool = Field(default=True, description="Whether to apply cross-encoder reranking")
