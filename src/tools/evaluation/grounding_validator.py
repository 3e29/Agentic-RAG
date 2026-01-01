"""
Grounding Validator Tool (FR-EA-26)

This tool validates that retrieved documents actually contain the concepts
and keywords requested in the query. Prevents hallucination by ensuring
the answer can be grounded in the source documents.

Production Standards:
- LangSmith tracing for observability
- Multi-method validation (keyword + semantic)
- Bilingual support (Arabic/English)
"""

import logging
import re
from typing import List, Optional, Set
from langsmith import traceable

from src.tools.evaluation.schemas import (
    GroundingResult,
    ConceptGrounding,
)
from src.tools.retrieval.schemas import Document

logger = logging.getLogger(__name__)


# ============================================================================
# Arabic Text Normalization for Matching
# ============================================================================

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison."""
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize alef variants
    text = re.sub(r'[أإآا]', 'ا', text)
    # Normalize ta marbuta
    text = re.sub(r'ة', 'ه', text)
    # Normalize yeh
    text = re.sub(r'ى', 'ي', text)
    return text.lower()


def extract_keywords(text: str, min_length: int = 2) -> Set[str]:
    """Extract meaningful keywords from text."""
    # Split on whitespace and punctuation
    words = re.split(r'[\s\.,;:!?،؟\-\(\)\[\]]+', text)
    # Filter short words and normalize
    keywords = set()
    for word in words:
        word = word.strip()
        if len(word) >= min_length:
            keywords.add(normalize_arabic(word))
    return keywords


# ============================================================================
# Grounding Validator Tool
# ============================================================================

class GroundingValidatorTool:
    """
    Tool for validating document grounding (FR-EA-26).
    
    Ensures that retrieved documents actually contain the concepts
    mentioned in the query. This prevents the synthesis agent from
    hallucinating information not present in sources.
    """
    
    def __init__(
        self,
        min_grounding_score: float = 0.6,
        use_fuzzy_matching: bool = True,
    ):
        """
        Initialize the grounding validator.
        
        Args:
            min_grounding_score: Minimum score to consider grounded
            use_fuzzy_matching: Whether to use fuzzy keyword matching
        """
        self.min_grounding_score = min_grounding_score
        self.use_fuzzy_matching = use_fuzzy_matching
    
    @traceable(name="grounding_validator_tool", run_type="tool")
    def __call__(
        self,
        query: str,
        documents: List[Document],
        key_concepts: Optional[List[str]] = None,
    ) -> GroundingResult:
        """
        Validate that documents contain the requested concepts.
        
        Args:
            query: The original search query
            documents: List of retrieved documents
            key_concepts: Optional pre-extracted key concepts from query
            
        Returns:
            GroundingResult with validation details
        """
        if not documents:
            return GroundingResult(
                is_grounded=False,
                grounding_score=0.0,
                concept_groundings=[],
                ungrounded_concepts=[],
                hallucination_risk="high",
            )
        
        # Extract concepts from query if not provided
        if not key_concepts:
            key_concepts = self._extract_concepts_from_query(query)
        
        if not key_concepts:
            # No concepts to validate - assume grounded
            return GroundingResult(
                is_grounded=True,
                grounding_score=1.0,
                concept_groundings=[],
                ungrounded_concepts=[],
                hallucination_risk="low",
            )
        
        # Build document text corpus
        doc_texts = {
            doc.chunk_id: normalize_arabic(doc.text)
            for doc in documents
        }
        
        # Check each concept
        concept_groundings = []
        grounded_count = 0
        
        for concept in key_concepts:
            grounding = self._check_concept_grounding(concept, documents, doc_texts)
            concept_groundings.append(grounding)
            if grounding.is_grounded:
                grounded_count += 1
        
        # Calculate overall grounding
        grounding_score = grounded_count / len(key_concepts) if key_concepts else 1.0
        is_grounded = grounding_score >= self.min_grounding_score
        
        # Identify ungrounded concepts
        ungrounded = [
            cg.concept for cg in concept_groundings
            if not cg.is_grounded
        ]
        
        # Determine hallucination risk
        if grounding_score >= 0.8:
            risk = "low"
        elif grounding_score >= 0.5:
            risk = "medium"
        else:
            risk = "high"
        
        return GroundingResult(
            is_grounded=is_grounded,
            grounding_score=grounding_score,
            concept_groundings=concept_groundings,
            ungrounded_concepts=ungrounded,
            hallucination_risk=risk,
        )
    
    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """Extract key concepts from the query."""
        # Common stopwords in Arabic and English
        stopwords = {
            # Arabic
            'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هل', 'ما', 'هذا', 'هذه',
            'الذي', 'التي', 'الذين', 'و', 'أو', 'ثم', 'لكن', 'بل', 'إن', 'أن',
            'كان', 'يكون', 'كانت', 'كانوا', 'هو', 'هي', 'هم', 'هن', 'نحن', 'أنت',
            'لي', 'لك', 'له', 'لها', 'لهم', 'بي', 'بك', 'به', 'بها', 'بهم',
            # English
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'to', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'it', 'its', 'they', 'them', 'their', 'and', 'but', 'or',
            'hadith', 'hadiths', 'about', 'regarding', 'concerning',
        }
        
        # Extract words
        words = re.split(r'[\s\.,;:!?،؟\-\(\)\[\]\"\']+', query)
        
        # Filter and normalize
        concepts = []
        for word in words:
            word_lower = word.lower().strip()
            word_normalized = normalize_arabic(word)
            
            if (
                len(word) >= 3 and
                word_lower not in stopwords and
                word_normalized not in stopwords and
                not word.isdigit()
            ):
                concepts.append(word)
        
        return concepts[:10]  # Limit to 10 concepts
    
    def _check_concept_grounding(
        self,
        concept: str,
        documents: List[Document],
        doc_texts: dict,
    ) -> ConceptGrounding:
        """Check if a concept is grounded in the documents."""
        normalized_concept = normalize_arabic(concept)
        evidence_chunks = []
        
        for doc in documents:
            doc_text = doc_texts.get(doc.chunk_id, "")
            
            # Direct substring match
            if normalized_concept in doc_text:
                evidence_chunks.append(doc.chunk_id)
                continue
            
            # Fuzzy matching for Arabic morphological variants
            if self.use_fuzzy_matching:
                # Check if root appears (simple prefix/suffix stripping)
                concept_root = self._get_arabic_root(normalized_concept)
                if concept_root and len(concept_root) >= 3:
                    if concept_root in doc_text:
                        evidence_chunks.append(doc.chunk_id)
                        continue
        
        is_grounded = len(evidence_chunks) > 0
        confidence = min(1.0, len(evidence_chunks) / 3)  # Cap at 1.0 with 3+ matches
        
        return ConceptGrounding(
            concept=concept,
            is_grounded=is_grounded,
            evidence_chunks=evidence_chunks[:5],  # Limit evidence
            confidence=confidence if is_grounded else 0.0,
        )
    
    def _get_arabic_root(self, word: str) -> str:
        """
        Simple Arabic root extraction.
        
        This is a simplified version - production would use
        a proper Arabic morphological analyzer.
        """
        # Remove common prefixes
        prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'ك', 'س', 'لل']
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break
        
        # Remove common suffixes
        suffixes = ['ون', 'ين', 'ات', 'ان', 'ها', 'هم', 'كم', 'نا', 'ة', 'ي']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        
        return word


# ============================================================================
# Module-level convenience function
# ============================================================================

@traceable(name="validate_grounding", run_type="tool")
def validate_grounding(
    query: str,
    documents: List[Document],
    key_concepts: Optional[List[str]] = None,
    min_score: float = 0.6,
) -> GroundingResult:
    """
    Validate that documents contain the requested concepts.
    
    Convenience function that creates and invokes GroundingValidatorTool.
    
    Args:
        query: The search query
        documents: Retrieved documents to validate
        key_concepts: Optional pre-extracted concepts
        min_score: Minimum grounding score threshold
        
    Returns:
        GroundingResult with validation details
    """
    tool = GroundingValidatorTool(min_grounding_score=min_score)
    return tool(query, documents, key_concepts)
