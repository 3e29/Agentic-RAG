"""
Quality Assessment Tool (FR-EA-22)

This tool assesses whether retrieved documents actually answer the user's query.
Uses an LLM to evaluate semantic relevance, not just keyword matching.

Production Standards:
- LangSmith tracing for observability
- Graceful fallback for LLM failures
- Bilingual support (Arabic/English)
"""

import logging
from typing import List, Optional
from langsmith import traceable

from src.tools.evaluation.schemas import (
    QualityAssessment,
    DocumentRelevance,
    RelevanceLevel,
)
from src.tools.retrieval.schemas import Document
from src.utils.llm_helper import call_llm_sync, parse_json_response

logger = logging.getLogger(__name__)


# ============================================================================
# Quality Assessment Prompt
# ============================================================================

QUALITY_ASSESSMENT_SYSTEM = """You are a quality assessment expert for a Hadith search system.

TASK: Evaluate whether the retrieved documents actually answer the user's query.

You will receive:
1. The original query (may be in Arabic or English)
2. A list of retrieved documents with their text

ASSESSMENT STEPS:
1. Extract KEY CONCEPTS from the query (topics, narrators, conditions, etc.)
2. For each document, assess how relevant it is to the query
3. Determine if the documents COLLECTIVELY answer the query

RELEVANCE LEVELS:
- high: Document directly addresses the query's main topic
- medium: Document is related but not directly answering
- low: Document has tangential relevance
- irrelevant: Document does not relate to the query

QUALITY THRESHOLDS:
- overall_score >= 0.7 + at least 2 high-relevance docs = is_sufficient: true
- overall_score < 0.5 OR no high-relevance docs = is_sufficient: false

NEGATIVE CONSTRAINTS:
- Do NOT output anything except the JSON
- Do NOT include markdown code blocks
- Output ONLY valid JSON

OUTPUT FORMAT:
{
    "overall_score": <0.0-1.0>,
    "is_sufficient": <true/false>,
    "reasoning": "<brief explanation>",
    "query_concepts": ["concept1", "concept2", ...],
    "concepts_covered": ["concept1", ...],
    "concepts_missing": ["concept2", ...],
    "document_assessments": [
        {
            "chunk_id": "<id>",
            "relevance": "high|medium|low|irrelevant",
            "relevance_score": <0.0-1.0>,
            "reason": "<why this relevance>",
            "key_concepts_matched": ["concept1", ...]
        }
    ]
}"""

QUALITY_ASSESSMENT_PROMPT = """QUERY: {query}

RETRIEVED DOCUMENTS ({doc_count} total):
{documents}

Assess the quality of these results for answering the query.
Output JSON only:"""


# ============================================================================
# Quality Assessment Tool
# ============================================================================

class QualityAssessmentTool:
    """
    Tool for assessing retrieval quality (FR-EA-22).
    
    Uses an LLM to semantically evaluate whether documents answer the query.
    Returns structured assessment with per-document relevance.
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the quality assessment tool.
        
        Args:
            use_llm: Whether to use LLM for assessment (False = heuristic only)
        """
        self.use_llm = use_llm
    
    @traceable(name="quality_assessment_tool", run_type="tool")
    def __call__(
        self,
        query: str,
        documents: List[Document],
        min_relevance_threshold: float = 0.5,
    ) -> QualityAssessment:
        """
        Assess the quality of retrieved documents.
        
        Args:
            query: The original search query
            documents: List of retrieved documents
            min_relevance_threshold: Minimum score to consider relevant
            
        Returns:
            QualityAssessment with overall score and per-doc assessments
        """
        if not documents:
            return QualityAssessment(
                overall_score=0.0,
                is_sufficient=False,
                reasoning="No documents retrieved",
                document_assessments=[],
                query_concepts=[],
                concepts_covered=[],
                concepts_missing=[],
            )
        
        if self.use_llm:
            return self._llm_assessment(query, documents, min_relevance_threshold)
        else:
            return self._heuristic_assessment(query, documents, min_relevance_threshold)
    
    def _llm_assessment(
        self,
        query: str,
        documents: List[Document],
        min_threshold: float,
    ) -> QualityAssessment:
        """Use LLM to assess quality."""
        # Format documents for prompt
        doc_texts = []
        for i, doc in enumerate(documents[:10]):  # Limit to 10 docs for token efficiency
            doc_texts.append(
                f"[Doc {i+1}] ID: {doc.chunk_id}\n"
                f"Collection: {doc.collection}\n"
                f"Text: {doc.text[:500]}..."
            )
        
        prompt = QUALITY_ASSESSMENT_PROMPT.format(
            query=query,
            doc_count=len(documents),
            documents="\n\n".join(doc_texts),
        )
        
        try:
            response = call_llm_sync(
                prompt=prompt,
                system_message=QUALITY_ASSESSMENT_SYSTEM,
                temperature=0.0,
                max_tokens=1500,
                metadata={"tool": "quality_assessment"},
            )
            
            parsed = parse_json_response(response)
            
            # Build document assessments
            doc_assessments = []
            for assessment in parsed.get("document_assessments", []):
                doc_assessments.append(DocumentRelevance(
                    chunk_id=assessment.get("chunk_id", ""),
                    relevance=RelevanceLevel(assessment.get("relevance", "low")),
                    relevance_score=float(assessment.get("relevance_score", 0.5)),
                    reason=assessment.get("reason", ""),
                    key_concepts_matched=assessment.get("key_concepts_matched", []),
                ))
            
            return QualityAssessment(
                overall_score=float(parsed.get("overall_score", 0.5)),
                is_sufficient=parsed.get("is_sufficient", False),
                reasoning=parsed.get("reasoning", "Assessment complete"),
                document_assessments=doc_assessments,
                query_concepts=parsed.get("query_concepts", []),
                concepts_covered=parsed.get("concepts_covered", []),
                concepts_missing=parsed.get("concepts_missing", []),
            )
            
        except Exception as e:
            logger.warning(f"LLM quality assessment failed: {e}, using heuristic")
            return self._heuristic_assessment(query, documents, min_threshold)
    
    def _heuristic_assessment(
        self,
        query: str,
        documents: List[Document],
        min_threshold: float,
    ) -> QualityAssessment:
        """Fallback heuristic-based assessment."""
        # Use retrieval scores as proxy for relevance
        high_score_docs = [d for d in documents if d.score >= 0.7]
        medium_score_docs = [d for d in documents if 0.4 <= d.score < 0.7]
        
        # Simple quality score based on score distribution
        if documents:
            avg_score = sum(d.score for d in documents) / len(documents)
            max_score = max(d.score for d in documents)
            overall_score = (avg_score * 0.6) + (max_score * 0.4)
        else:
            overall_score = 0.0
        
        is_sufficient = (
            overall_score >= min_threshold and
            len(high_score_docs) >= 1
        )
        
        # Build basic assessments
        doc_assessments = []
        for doc in documents[:10]:
            if doc.score >= 0.7:
                relevance = RelevanceLevel.HIGH
            elif doc.score >= 0.4:
                relevance = RelevanceLevel.MEDIUM
            elif doc.score >= 0.2:
                relevance = RelevanceLevel.LOW
            else:
                relevance = RelevanceLevel.IRRELEVANT
            
            doc_assessments.append(DocumentRelevance(
                chunk_id=doc.chunk_id,
                relevance=relevance,
                relevance_score=doc.score,
                reason=f"Based on retrieval score: {doc.score:.2f}",
                key_concepts_matched=[],
            ))
        
        return QualityAssessment(
            overall_score=overall_score,
            is_sufficient=is_sufficient,
            reasoning=f"Heuristic: {len(high_score_docs)} high-score, {len(medium_score_docs)} medium-score docs",
            document_assessments=doc_assessments,
            query_concepts=[],
            concepts_covered=[],
            concepts_missing=[],
        )


# ============================================================================
# Module-level convenience function
# ============================================================================

@traceable(name="assess_quality", run_type="tool")
def assess_quality(
    query: str,
    documents: List[Document],
    use_llm: bool = True,
) -> QualityAssessment:
    """
    Assess the quality of retrieved documents for a query.
    
    Convenience function that creates and invokes QualityAssessmentTool.
    
    Args:
        query: The search query
        documents: Retrieved documents to assess
        use_llm: Whether to use LLM (True) or heuristics (False)
        
    Returns:
        QualityAssessment with detailed evaluation
    """
    tool = QualityAssessmentTool(use_llm=use_llm)
    return tool(query, documents)
