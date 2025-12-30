"""
Filter and Expansion Tools for Hadith RAG System

Implements metadata extraction and query expansion:
1. MetadataFilterTool (FR-RA-14) - Convert NL constraints to DB filters
2. QueryExpansionTool - Generate synonyms and translations

Production Standards:
- LLM-based extraction with fallback patterns
- Arabic/English synonym dictionaries
- Graceful error handling
"""

import logging
import re
from typing import Dict, List, Optional, Any
from langsmith import traceable

from src.tools.retrieval.schemas import (
    MetadataFilter,
    ExpandedQuery,
)
from src.utils.llm_helper import call_llm_sync, parse_json_response, LLMError
from src.utils.prompts import format_prompt

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Pattern-based Extraction (Fallback)
# ============================================================================

# Collection patterns
BUKHARI_PATTERN = re.compile(
    r'\b(bukhari|البخاري|صحيح البخاري|sahih al-bukhari)\b', 
    re.IGNORECASE | re.UNICODE
)
MUSLIM_PATTERN = re.compile(
    r'\b(muslim|مسلم|صحيح مسلم|sahih muslim)\b', 
    re.IGNORECASE | re.UNICODE
)

# Hadith number patterns
HADITH_NUMBER_PATTERN = re.compile(
    r'(?:hadith|حديث)?\s*(?:number|رقم|#|no\.?)?\s*(\d+)',
    re.IGNORECASE | re.UNICODE
)

# Book patterns
BOOK_PATTERN = re.compile(
    r'(?:book|كتاب)\s+(?:of\s+)?(?:the\s+)?([a-zA-Z\u0600-\u06FF]+)',
    re.IGNORECASE | re.UNICODE
)

# Narrator patterns
NARRATOR_PATTERNS = [
    re.compile(r"narrated by\s+([^,\.]+)", re.IGNORECASE),
    re.compile(r"رواه\s+([^,\.]+)", re.UNICODE),
    re.compile(r"عن\s+(أبو\s+\w+|أنس|عائشة|ابن عباس|ابن عمر)", re.UNICODE),
    re.compile(r"(Abu\s+\w+|Anas|Aisha|Ibn\s+\w+)", re.IGNORECASE),
]





# ============================================================================
# Tool Classes
# ============================================================================

class MetadataFilterTool:
    """
    Convert natural language constraints to database filters (FR-RA-14).
    
    Uses LLM for complex extraction with regex fallback.
    """
    
    name: str = "metadata_filter"
    description: str = "Extract structured filters from natural language query"
    
    @traceable(name="metadata_filter_tool")
    def __call__(self, query: str, use_llm: bool = True) -> MetadataFilter:
        """Extract metadata filters from query."""
        return extract_metadata_filters(query, use_llm=use_llm)


class QueryExpansionTool:
    """
    Generate query expansions with synonyms and translations.
    
    Produces 3-5 alternative queries for better recall.
    """
    
    name: str = "query_expansion"
    description: str = "Expand query with synonyms and translations"
    
    @traceable(name="query_expansion_tool")
    def __call__(self, query: str, use_llm: bool = True) -> ExpandedQuery:
        """Expand query with alternatives."""
        return expand_query(query, use_llm=use_llm)


# ============================================================================
# Functional Implementations
# ============================================================================

@traceable(name="extract_metadata_filters")
def extract_metadata_filters(query: str, use_llm: bool = True) -> MetadataFilter:
    """
    Extract structured metadata filters from natural language query.
    
    Args:
        query: User's natural language query
        use_llm: Whether to use LLM for extraction (True) or just patterns
        
    Returns:
        MetadataFilter with extracted constraints
    """
    logger.info(f"Extracting filters from: '{query[:100]}...'")
    
    # Start with pattern-based extraction
    filters = _extract_filters_pattern(query)
    
    # Enhance with LLM if enabled and query is complex
    if use_llm and _is_complex_query(query):
        try:
            llm_filters = _extract_filters_llm(query)
            filters = _merge_filters(filters, llm_filters)
        except Exception as e:
            logger.warning(f"LLM filter extraction failed, using patterns: {e}")
    
    logger.info(f"Extracted filters: {filters.model_dump(exclude_none=True)}")
    return filters


def _extract_filters_pattern(query: str) -> MetadataFilter:
    """Extract filters using regex patterns."""
    collection = None
    book_id = None
    chapter_id = None
    narrator = None
    hadith_id_in_book = None  # User-facing hadith number (not internal hadith_id)
    
    # Collection detection
    if BUKHARI_PATTERN.search(query):
        collection = "bukhari"
    elif MUSLIM_PATTERN.search(query):
        collection = "muslim"
    
    # Hadith number - extract as hadith_id_in_book (the number users reference)
    hadith_match = HADITH_NUMBER_PATTERN.search(query)
    if hadith_match:
        hadith_id_in_book = int(hadith_match.group(1))
    
    # Narrator detection
    for pattern in NARRATOR_PATTERNS:
        match = pattern.search(query)
        if match:
            narrator = match.group(1).strip()
            break
    
    return MetadataFilter(
        collection=collection,
        book_id=book_id,
        chapter_id=chapter_id,
        narrator=narrator,
        hadith_id_in_book=hadith_id_in_book,  # User-facing number
        raw_constraints=query,
    )


def _extract_filters_llm(query: str) -> MetadataFilter:
    """
    Extract filters using LLM for complex queries.
    
    Uses centralized prompts from src/utils/prompts.py.
    """
    # Get prompt from centralized prompts (includes temperature=0.0 and chapter_title_en)
    system_message, prompt, temperature, max_tokens = format_prompt(
        "retrieval", "metadata_extraction", query=query
    )

    try:
        response = call_llm_sync(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,  # 0.0 from prompts.py
            max_tokens=max_tokens,
        )
        
        parsed = parse_json_response(response)
        return MetadataFilter(**parsed)
        
    except Exception as e:
        logger.error(f"LLM filter extraction failed: {e}")
        raise


def _is_complex_query(query: str) -> bool:
    """Determine if query needs LLM extraction."""
    # Simple heuristics for complexity
    word_count = len(query.split())
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
    has_specific_refs = bool(re.search(r'book|chapter|كتاب|باب|\d+', query, re.IGNORECASE))
    
    return word_count > 5 or (has_arabic and has_specific_refs)


def _merge_filters(pattern_filters: MetadataFilter, llm_filters: MetadataFilter) -> MetadataFilter:
    """Merge pattern and LLM extracted filters, preferring LLM when confident."""
    merged = pattern_filters.model_copy()
    
    # Only use LLM values if they have high confidence
    if llm_filters.confidence >= 0.7:
        if llm_filters.collection and not merged.collection:
            merged.collection = llm_filters.collection
        if llm_filters.book_id is not None and merged.book_id is None:
            merged.book_id = llm_filters.book_id
        if llm_filters.chapter_id is not None and merged.chapter_id is None:
            merged.chapter_id = llm_filters.chapter_id
        if llm_filters.chapter_title_en and not merged.chapter_title_en:
            merged.chapter_title_en = llm_filters.chapter_title_en
        if llm_filters.chapter_title_ar and not merged.chapter_title_ar:
            merged.chapter_title_ar = llm_filters.chapter_title_ar
        if llm_filters.narrator and not merged.narrator:
            merged.narrator = llm_filters.narrator
        # Prefer hadith_id_in_book (user-facing) over internal hadith_id
        if llm_filters.hadith_id_in_book is not None and merged.hadith_id_in_book is None:
            merged.hadith_id_in_book = llm_filters.hadith_id_in_book
        if llm_filters.hadith_id is not None and merged.hadith_id is None:
            merged.hadith_id = llm_filters.hadith_id
        if llm_filters.language and not merged.language:
            merged.language = llm_filters.language
    
    return merged


@traceable(name="expand_query")
def expand_query(query: str, use_llm: bool = True) -> ExpandedQuery:
    """
    Expand query with synonyms and translations.
    
    Args:
        query: Original user query
        use_llm: Whether to use LLM for expansion
        
    Returns:
        ExpandedQuery with alternative search terms
    """
    logger.info(f"Expanding query: '{query[:100]}...'")
    
    expanded_terms = []
    translations = {}
    
    # LLM-based expansion
    if use_llm:
        try:
            llm_expansions = _expand_query_llm(query)
            expanded_terms.extend(llm_expansions.get("terms", []))
            translations.update(llm_expansions.get("translations", {}))
        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")
    
    # Deduplicate
    expanded_terms = list(set(expanded_terms))
    
    result = ExpandedQuery(
        original_query=query,
        expanded_terms=expanded_terms[:5],  # Limit to 5 expansions
        translations=translations,
        confidence=0.8 if expanded_terms else 0.5,
    )
    
    logger.info(f"Query expanded to {len(result.get_all_queries())} variants")
    return result


def _expand_query_llm(query: str) -> Dict[str, Any]:
    """Use LLM to generate query expansions."""
    system_message = """You are a query expansion expert for Islamic Hadith search.

Generate 3-5 alternative search terms/phrases that would help find relevant hadiths.

Guidelines:
- Include Arabic equivalents for English terms
- Include English equivalents for Arabic terms  
- Add common synonyms and related concepts
- Keep expansions relevant to Islamic scholarship
- For Arabic terms, include shadda (ّ) diacritic where linguistically correct

Respond ONLY with valid JSON:
{
  "terms": ["term1", "term2", "term3"],
  "translations": {"original_term": "translated_term"}
}"""

    prompt = f'Expand this hadith search query:\n\n"{query}"\n\nRespond ONLY with JSON.'

    try:
        response = call_llm_sync(
            prompt=prompt,
            system_message=system_message,
            temperature=0.45,  # Deterministic for consistent results
            max_tokens=256,
        )
        
        parsed = parse_json_response(response)
        return {
            "terms": parsed.get("terms", []),
            "translations": parsed.get("translations", {}),
        }
        
    except Exception as e:
        logger.error(f"LLM query expansion failed: {e}")
        raise
