"""
Query Processing Tools for Hadith RAG System

This module implements the query analysis tools following Clean Architecture principles:

**Pre-processing Tools (No LLM):**
1. Query Normalization Tool - Arabic text normalization via regex
2. Collection Target Detection Tool - Identify which hadith collections to search

**LLM-based Tools:**
3. Input Source Identification Tool - Classify if user queries DB or provides own text
4. Typo Correction Tool (FR-QAA-05) - Handles Arabic/English spelling and diacritics
5. Intent Classification Tool (FR-QAA-07) - Categorizes query intent
6. Query Decomposition Tool (FR-QAA-08) - Breaks complex queries into sub-queries

Production Standards:
- Type safety with typing and Pydantic v2
- SOLID principles (Single Responsibility per tool)
- Defensive programming with try/except and fallback values
- Enums over strings for fixed categories
- Compiled regex patterns at module level for performance
- Full observability via logging and LangSmith tracing

**v2.0 Updates:**
- Centralized prompts from src/utils/prompts.py
- temperature=0.0 for all JSON-outputting tasks
- Few-shot examples for better consistency
"""

import logging
import re
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from langsmith import traceable

from src.utils.llm_helper import call_llm_sync, parse_json_response, LLMError
from src.utils.prompts import format_prompt, QUERY_ANALYSIS_PROMPTS

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums for Type Safety (SOLID: Avoid Magic Strings)
# ============================================================================

class InputSource(str, Enum):
    """Source type for user input classification."""
    BASE_KNOWLEDGE = "base_knowledge"
    USER_TEXT = "user_text"
    FILE_UPLOAD = "file_upload"


class CollectionTarget(str, Enum):
    """Target hadith collection for search."""
    BUKHARI = "bukhari"
    MUSLIM = "muslim"


class QueryIntent(str, Enum):
    """Classified intent of the query."""
    THEMATIC_SEARCH = "thematic_search"
    SPECIFIC_LOOKUP = "specific_lookup"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    METADATA_QUERY = "metadata_query"


# ============================================================================
# Compiled Regex Patterns (Module-level for Performance)
# ============================================================================

# Arabic normalization patterns - compiled once at module load
ALIF_PATTERN = re.compile(r'[أإآٱ]')
TASHKEEL_PATTERN = re.compile(r'[\u064B-\u0652\u0670]')
TATWEEL_PATTERN = re.compile(r'ـ+')
TEH_MARBUTA_PATTERN = re.compile(r'ة')
WHITESPACE_PATTERN = re.compile(r'\s+')

# Collection detection patterns
BUKHARI_PATTERN = re.compile(r'\b(bukhari|البخاري|صحيح البخاري)\b', re.IGNORECASE)
MUSLIM_PATTERN = re.compile(r'\b(muslim|مسلم|صحيح مسلم)\b', re.IGNORECASE)

# Input source detection patterns
USER_TEXT_PATTERNS = [
    re.compile(r'(explain|analyze|interpret)\s+(this|the following|my)\s+(text|hadith|passage|file)', re.IGNORECASE),
    re.compile(r'here\s+is\s+(a|the|my)\s+(text|hadith|passage|file)', re.IGNORECASE),
    re.compile(r':\s*["\'].*["\']', re.IGNORECASE),
    re.compile(r'(this|the following)\s+(text|hadith|says?|file)', re.IGNORECASE),
]


# ============================================================================
# Pydantic Models for Structured Output (Type Safety)
# ============================================================================

class QueryNormalizationOutput(BaseModel):
    """Structured output for query normalization (no LLM)."""
    
    normalized_text: str = Field(description="The normalized text with Arabic characters standardized")
    original_text: str = Field(description="The original input text before normalization")
    transformations_applied: List[str] = Field(default_factory=list, description="List of normalization transformations applied")
    
    @field_validator('normalized_text')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Normalized text cannot be empty")
        return v.strip()


class InputSourceOutput(BaseModel):
    """Structured output for input source identification."""
    
    source_type: Literal["base_knowledge", "user_text", "file_upload"] = Field(description="Classified source type of the input")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the classification")
    reasoning: str = Field(description="Brief explanation of the classification decision")


class CollectionTargetOutput(BaseModel):
    """Structured output for collection target detection."""
    
    targets: List[Literal["bukhari", "muslim"]] = Field(description="List of target collections to search")
    reasoning: str = Field(description="Explanation of why these collections were selected")


class TypoCorrectionOutput(BaseModel):
    """Structured output for typo correction."""
    
    corrected_text: str = Field(description="The corrected text with typos fixed")
    language: Literal["ar", "en", "mixed"] = Field(description="Detected dominant language of the input text")
    desired_output_language: Literal["arabic", "english"] = Field(description="User's preferred language for results (explicit preference or inferred from query language)")
    corrections_made: List[str] = Field(default_factory=list, description="List of corrections applied")
    
    @field_validator('corrected_text')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Corrected text cannot be empty")
        return v.strip()
    
    @field_validator('language', mode='before')
    @classmethod
    def normalize_language(cls, v: str) -> str:
        """Normalize language values - LLM sometimes returns 'english' instead of 'en'."""
        if v in ['english', 'English', 'EN']:
            return 'en'
        if v in ['arabic', 'Arabic', 'AR']:
            return 'ar'
        return v


class IntentClassificationOutput(BaseModel):
    """Structured output for intent classification."""
    
    intent: Literal["thematic_search", "specific_lookup", "comparative_analysis", "metadata_query"] = Field(description="Classified intent of the query")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the classification")
    reasoning: str = Field(description="Brief explanation of why this intent was chosen")


class QueryDecompositionOutput(BaseModel):
    """Structured output for query decomposition."""
    
    is_complex: bool = Field(description="Whether the query is complex and needs decomposition")
    sub_queries: List[str] = Field(default_factory=list, description="List of atomic sub-queries")
    reasoning: str = Field(description="Explanation of the decomposition strategy")
    
    @field_validator('sub_queries')
    @classmethod
    def validate_sub_queries(cls, v: List[str], info) -> List[str]:
        is_complex = info.data.get('is_complex', False)
        if is_complex and len(v) == 0:
            raise ValueError("Complex queries must have at least one sub-query")
        return v


# ============================================================================
# Tool 1: Query Normalization Tool (Pure Python/Regex - No LLM)
# ============================================================================

@traceable(name="query_normalization_tool")
def query_normalization_tool(query: str) -> QueryNormalizationOutput:
    """
    Normalize Arabic text for improved search matching (NO LLM).
    
    Operations:
    1. Normalize Alif variants (أ, إ, آ, ٱ) -> ا
    2. Remove Tashkeel (diacritical marks)
    3. Remove Tatweel (kashida elongation)
    4. Normalize Teh Marbuta: ة -> ه
    5. Normalize whitespace
    """
    logger.info(f"Running query normalization on: {query[:100]}...")
    
    transformations = []
    normalized = query
    
    try:
        if ALIF_PATTERN.search(normalized):
            normalized = ALIF_PATTERN.sub('ا', normalized)
            transformations.append("Normalized Alif variants (أإآٱ → ا)")
        
        if TASHKEEL_PATTERN.search(normalized):
            normalized = TASHKEEL_PATTERN.sub('', normalized)
            transformations.append("Removed Tashkeel (diacritical marks)")
        
        if TATWEEL_PATTERN.search(normalized):
            normalized = TATWEEL_PATTERN.sub('', normalized)
            transformations.append("Removed Tatweel (ـ elongation)")
        
        if TEH_MARBUTA_PATTERN.search(normalized):
            normalized = TEH_MARBUTA_PATTERN.sub('ه', normalized)
            transformations.append("Normalized Teh Marbuta (ة → ه)")
        
        original_len = len(normalized)
        normalized = WHITESPACE_PATTERN.sub(' ', normalized).strip()
        if len(normalized) != original_len:
            transformations.append("Normalized whitespace")
        
        if not transformations:
            transformations.append("No normalization needed")
        
        logger.info(f"Query normalization complete. {len(transformations)} transformation(s) applied.")
        
        return QueryNormalizationOutput(
            normalized_text=normalized,
            original_text=query,
            transformations_applied=transformations
        )
        
    except Exception as e:
        logger.error(f"Query normalization failed: {e}")
        return QueryNormalizationOutput(
            normalized_text=query,
            original_text=query,
            transformations_applied=["Error occurred, no normalization applied"]
        )


# ============================================================================
# Tool 2: Collection Target Detection Tool (Keyword-based - No LLM)
# ============================================================================

@traceable(name="collection_target_detection_tool")
def collection_target_detection_tool(query: str) -> CollectionTargetOutput:
    """
    Detect which hadith collection(s) the user wants to search (NO LLM).
    
    Uses keyword-based detection. If no explicit mention, returns all collections.
    """
    logger.info(f"Detecting collection targets for: {query[:100]}...")
    
    targets: List[str] = []
    reasoning_parts: List[str] = []
    
    try:
        bukhari_match = BUKHARI_PATTERN.search(query)
        muslim_match = MUSLIM_PATTERN.search(query)
        
        if bukhari_match:
            targets.append("bukhari")
            reasoning_parts.append(f"Bukhari mentioned: '{bukhari_match.group()}'")
            logger.info("Detected explicit Bukhari collection reference")
        
        if muslim_match:
            targets.append("muslim")
            reasoning_parts.append(f"Muslim mentioned: '{muslim_match.group()}'")
            logger.info("Detected explicit Muslim collection reference")
        
        if targets:
            return CollectionTargetOutput(targets=targets, reasoning="; ".join(reasoning_parts))
        
        logger.info("No specific collection mentioned, targeting all collections")
        return CollectionTargetOutput(
            targets=["bukhari", "muslim"],
            reasoning="No specific collection mentioned, searching all available collections"
        )
        
    except Exception as e:
        logger.error(f"Collection target detection failed: {e}")
        return CollectionTargetOutput(
            targets=["bukhari", "muslim"],
            reasoning="Detection failed, defaulting to all collections"
        )


# ============================================================================
# Tool 3: Input Source Identification Tool (LLM-based)
# ============================================================================

@traceable(name="input_source_identification_tool")
def input_source_identification_tool(query: str) -> InputSourceOutput:
    """
    Identify whether user is querying the database or providing their own text.
    
    Categories:
    - base_knowledge: User is asking a question to search the hadith database
    - user_text: User is providing their own text for analysis
    - file_upload: User has uploaded a file
    
    Uses centralized prompts with Few-Shot examples.
    """
    logger.info(f"Identifying input source for: {query[:100]}...")
    
    # Quick regex-based detection first
    for pattern in USER_TEXT_PATTERNS:
        if pattern.search(query):
            logger.info("Input source identified as user_text via pattern matching")
            return InputSourceOutput(
                source_type="user_text",
                confidence=0.85,
                reasoning="Query contains patterns indicating user-provided text for analysis"
            )
            
    # Heuristic: If no user text patterns and query is short/medium, assume base_knowledge
    # This avoids LLM call for standard queries which are the vast majority
    if len(query.split()) < 50:
        logger.info("Input source identified as base_knowledge via heuristic (short query, no user text patterns)")
        return InputSourceOutput(
            source_type="base_knowledge",
            confidence=0.9,
            reasoning="Short query with no user text patterns"
        )
    
    # Get prompt from centralized prompts (includes temperature=0.0 and few-shot)
    system_message, prompt, temperature, max_tokens = format_prompt(
        "query_analysis", "input_source_identification", query=query
    )

    try:
        response = call_llm_sync(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,  # 0.0 from prompts.py
            max_tokens=max_tokens,
            metadata={"tool": "input_source_identification"}
        )
        
        parsed = parse_json_response(response)
        result = InputSourceOutput(**parsed)
        
        logger.info(f"Input source identified as: {result.source_type} (confidence: {result.confidence:.2f})")
        return result
        
    except Exception as e:
        logger.error(f"Input source identification failed: {e}")
        return InputSourceOutput(
            source_type="base_knowledge",
            confidence=0.5,
            reasoning="Classification failed, defaulting to base_knowledge query"
        )


# ============================================================================
# Tool 4: Typo Correction Tool (FR-QAA-05) - LLM-based
# ============================================================================

@traceable(name="typo_correction_tool")
def typo_correction_tool(query: str) -> TypoCorrectionOutput:
    """
    Detect and correct spelling errors in user queries.
    
    Handles both Arabic and English text, with special attention to Arabic diacritics.
    Uses centralized prompts from src/utils/prompts.py with temperature=0.0.
    """
    logger.info(f"Running typo correction on query: {query[:100]}...")
    
    # Get prompt from centralized prompts (includes temperature=0.0)
    system_message, prompt, temperature, max_tokens = format_prompt(
        "query_analysis", "typo_correction", query=query
    )
    
    # Helper to detect dominant language from query text
    def detect_dominant_language(text: str) -> str:
        """Detect if query is primarily Arabic or English."""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return "en"
        return "ar" if arabic_chars / total_alpha > 0.5 else "en"

    try:
        response = call_llm_sync(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,  # 0.0 from prompts.py
            max_tokens=max_tokens,
            metadata={"tool": "typo_correction", "query_length": len(query)}
        )
        
        parsed = parse_json_response(response)
        
        # Ensure desired_output_language is present (fallback to inferred language)
        if 'desired_output_language' not in parsed or parsed['desired_output_language'] is None:
            detected_lang = detect_dominant_language(query)
            parsed['desired_output_language'] = "arabic" if detected_lang == "ar" else "english"
        
        result = TypoCorrectionOutput(**parsed)
        
        logger.info(f"Typo correction completed. Language: {result.language}, Output: {result.desired_output_language}, Corrections: {len(result.corrections_made)}")
        return result
        
    except Exception as e:
        logger.error(f"Typo correction failed: {e}")
        # Fallback: detect language from query
        detected_lang = detect_dominant_language(query)
        return TypoCorrectionOutput(
            corrected_text=query,
            language=detected_lang,
            desired_output_language="arabic" if detected_lang == "ar" else "english",
            corrections_made=["Error occurred, no corrections applied"]
        )


# ============================================================================
# Tool 5: Intent Classification Tool (FR-QAA-07) - LLM-based
# ============================================================================

@traceable(name="intent_classification_tool")
def intent_classification_tool(query: str) -> IntentClassificationOutput:
    """
    Classify the query intent into one of three categories.
    
    Categories:
    - thematic_search: Broad conceptual queries (e.g., "hadiths about prayer")
    - specific_lookup: Looking for specific hadith(s) by narrator, book, number, or text
    - comparative_analysis: Comparing topics, narrators, or seeking relationships
    
    Uses centralized prompts with Few-Shot examples for consistent output.
    """
    logger.info(f"Classifying intent for query: {query[:100]}...")
    
    # Get prompt from centralized prompts (includes temperature=0.0 and few-shot examples)
    system_message, prompt, temperature, max_tokens = format_prompt(
        "query_analysis", "intent_classification", query=query
    )

    try:
        response = call_llm_sync(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,  # 0.0 from prompts.py
            max_tokens=max_tokens,
            metadata={"tool": "intent_classification", "query": query}
        )
        
        parsed = parse_json_response(response)
        result = IntentClassificationOutput(**parsed)
        
        logger.info(f"Intent classified as: {result.intent} (confidence: {result.confidence:.2f})")
        return result
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return IntentClassificationOutput(
            intent="thematic_search",
            confidence=0.5,
            reasoning="Classification failed, defaulting to thematic search"
        )


# ============================================================================
# Tool 6: Query Decomposition Tool (FR-QAA-08) - LLM-based
# ============================================================================

@traceable(name="query_decomposition_tool")
def query_decomposition_tool(query: str) -> QueryDecompositionOutput:
    """
    Break complex multi-part questions into atomic sub-queries.
    
    Only decompose queries with multiple retrieval aspects (topics, books, narrators).
    The system ONLY retrieves hadiths - never create sub-queries for explanations.
    
    Uses centralized prompts with Chain-of-Thought reasoning.
    """
    logger.info(f"Analyzing query complexity: {query[:100]}...")
    
    # Get prompt from centralized prompts (includes temperature=0.0 and CoT)
    system_message, prompt, temperature, max_tokens = format_prompt(
        "query_analysis", "query_decomposition", query=query
    )

    try:
        response = call_llm_sync(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,  # 0.0 from prompts.py
            max_tokens=max_tokens,
            metadata={"tool": "query_decomposition", "query": query}
        )
        
        parsed = parse_json_response(response)
        result = QueryDecompositionOutput(**parsed)
        
        logger.info(f"Query decomposition: is_complex={result.is_complex}, sub_queries_count={len(result.sub_queries)}")
        return result
        
    except Exception as e:
        logger.error(f"Query decomposition failed: {e}")
        return QueryDecompositionOutput(
            is_complex=False,
            sub_queries=[],
            reasoning="Decomposition failed, treating as simple query"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def get_query_processing_tools():
    """Get all query processing tools as a dict for LangGraph integration."""
    return {
        "normalization": query_normalization_tool,
        "input_source": input_source_identification_tool,
        "collection_target": collection_target_detection_tool,
        "typo_correction": typo_correction_tool,
        "intent_classification": intent_classification_tool,
        "query_decomposition": query_decomposition_tool,
    }


def get_preprocessing_tools():
    """Get tools that run without LLM calls (fast path)."""
    return {
        "normalization": query_normalization_tool,
        "collection_target": collection_target_detection_tool,
    }


def get_llm_tools():
    """Get tools that require LLM calls."""
    return {
        "input_source": input_source_identification_tool,
        "typo_correction": typo_correction_tool,
        "intent_classification": intent_classification_tool,
        "query_decomposition": query_decomposition_tool,
    }
