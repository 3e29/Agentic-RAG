"""
Gap Identification Tool (FR-EA-23)

This tool identifies what information is missing from the retrieved results.
Provides actionable feedback for retry strategies.

Production Standards:
- LangSmith tracing for observability
- Graceful fallback for LLM failures
- Bilingual support (Arabic/English)
"""

import logging
from typing import List, Optional
from langsmith import traceable

from src.tools.evaluation.schemas import (
    GapAnalysis,
    InformationGap,
    GapType,
)
from src.tools.retrieval.schemas import Document
from src.utils.llm_helper import call_llm_sync, parse_json_response

logger = logging.getLogger(__name__)


# ============================================================================
# Gap Identification Prompt
# ============================================================================

GAP_IDENTIFICATION_SYSTEM = """You are an information gap analyst for a Hadith search system.

TASK: Identify what information is MISSING from the search results relative to the query.

You will receive:
1. The original query (may be in Arabic or English)
2. A list of retrieved documents
3. (Optional) Quality assessment results

GAP TYPES:
- missing_topic: A key topic from the query is not covered at all
- incomplete: Topic is mentioned but not fully explained
- wrong_collection: Results are from the wrong hadith collection
- wrong_language: Results are in the wrong language
- insufficient: Not enough supporting context

SEVERITY SCALE (0.0 - 1.0):
- 1.0: Critical - completely missing core information
- 0.7: Major - important aspect missing
- 0.4: Moderate - would improve results
- 0.1: Minor - nice to have

SUGGESTED ACTIONS:
- "expand_query" - Add synonyms or related terms
- "relax_filters" - Remove restrictive filters
- "change_collection" - Search different collection
- "find_chapter" - Locate specific chapter first
- "keyword_search" - Try keyword-based search
- "none" - No action needed

NEGATIVE CONSTRAINTS:
- Do NOT output anything except the JSON
- Do NOT include markdown code blocks
- Output ONLY valid JSON

OUTPUT FORMAT:
{
    "has_gaps": <true/false>,
    "coverage_score": <0.0-1.0>,
    "primary_gap": "<most critical missing info or null>",
    "gaps": [
        {
            "gap_type": "missing_topic|incomplete|wrong_collection|wrong_language|insufficient",
            "description": "<what is missing>",
            "severity": <0.0-1.0>,
            "suggested_action": "<action to take>",
            "suggested_query": "<alternative query or null>"
        }
    ],
    "feedback_for_retry": "<actionable instruction if retry needed>"
}"""

GAP_IDENTIFICATION_PROMPT = """QUERY: {query}

RETRIEVED DOCUMENTS ({doc_count} total):
{documents}

QUALITY CONTEXT:
- Overall quality score: {quality_score}
- Concepts covered: {concepts_covered}
- Concepts missing: {concepts_missing}

Identify information gaps and suggest improvements.
Output JSON only:"""


# ============================================================================
# Gap Identification Tool
# ============================================================================

class GapIdentificationTool:
    """
    Tool for identifying information gaps (FR-EA-23).
    
    Analyzes what's missing from search results and provides
    actionable feedback for retry strategies.
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the gap identification tool.
        
        Args:
            use_llm: Whether to use LLM for analysis (False = heuristic only)
        """
        self.use_llm = use_llm
    
    @traceable(name="gap_identification_tool", run_type="tool")
    def __call__(
        self,
        query: str,
        documents: List[Document],
        quality_score: float = 0.5,
        concepts_covered: Optional[List[str]] = None,
        concepts_missing: Optional[List[str]] = None,
    ) -> GapAnalysis:
        """
        Identify gaps in the retrieved documents.
        
        Args:
            query: The original search query
            documents: List of retrieved documents
            quality_score: Overall quality score from assessment
            concepts_covered: Concepts found in results
            concepts_missing: Concepts not found in results
            
        Returns:
            GapAnalysis with identified gaps and suggestions
        """
        concepts_covered = concepts_covered or []
        concepts_missing = concepts_missing or []
        
        if not documents:
            return GapAnalysis(
                has_gaps=True,
                gaps=[
                    InformationGap(
                        gap_type=GapType.MISSING_TOPIC,
                        description="No documents retrieved at all",
                        severity=1.0,
                        suggested_action="expand_query",
                        suggested_query=None,
                    )
                ],
                coverage_score=0.0,
                primary_gap="No search results found",
                feedback_for_retry="Try expanding the query with synonyms or related terms",
            )
        
        if self.use_llm:
            return self._llm_analysis(
                query, documents, quality_score, concepts_covered, concepts_missing
            )
        else:
            return self._heuristic_analysis(
                query, documents, quality_score, concepts_covered, concepts_missing
            )
    
    def _llm_analysis(
        self,
        query: str,
        documents: List[Document],
        quality_score: float,
        concepts_covered: List[str],
        concepts_missing: List[str],
    ) -> GapAnalysis:
        """Use LLM to identify gaps."""
        # Format documents for prompt
        doc_texts = []
        for i, doc in enumerate(documents[:8]):  # Limit for token efficiency
            doc_texts.append(
                f"[Doc {i+1}] Collection: {doc.collection}, Score: {doc.score:.2f}\n"
                f"Text: {doc.text[:400]}..."
            )
        
        prompt = GAP_IDENTIFICATION_PROMPT.format(
            query=query,
            doc_count=len(documents),
            documents="\n\n".join(doc_texts),
            quality_score=f"{quality_score:.2f}",
            concepts_covered=", ".join(concepts_covered) if concepts_covered else "None identified",
            concepts_missing=", ".join(concepts_missing) if concepts_missing else "None identified",
        )
        
        try:
            response = call_llm_sync(
                prompt=prompt,
                system_message=GAP_IDENTIFICATION_SYSTEM,
                temperature=0.0,
                max_tokens=1000,
                metadata={"tool": "gap_identification"},
            )
            
            parsed = parse_json_response(response)
            
            # Build gaps list
            gaps = []
            for gap_data in parsed.get("gaps", []):
                try:
                    gaps.append(InformationGap(
                        gap_type=GapType(gap_data.get("gap_type", "incomplete")),
                        description=gap_data.get("description", ""),
                        severity=float(gap_data.get("severity", 0.5)),
                        suggested_action=gap_data.get("suggested_action", "none"),
                        suggested_query=gap_data.get("suggested_query"),
                    ))
                except ValueError:
                    # Invalid gap type, use incomplete as fallback
                    gaps.append(InformationGap(
                        gap_type=GapType.INCOMPLETE_COVERAGE,
                        description=gap_data.get("description", "Unknown gap"),
                        severity=float(gap_data.get("severity", 0.5)),
                        suggested_action=gap_data.get("suggested_action", "none"),
                        suggested_query=gap_data.get("suggested_query"),
                    ))
            
            return GapAnalysis(
                has_gaps=parsed.get("has_gaps", len(gaps) > 0),
                gaps=gaps,
                coverage_score=float(parsed.get("coverage_score", quality_score)),
                primary_gap=parsed.get("primary_gap"),
                feedback_for_retry=parsed.get("feedback_for_retry", ""),
            )
            
        except Exception as e:
            logger.warning(f"LLM gap analysis failed: {e}, using heuristic")
            return self._heuristic_analysis(
                query, documents, quality_score, concepts_covered, concepts_missing
            )
    
    def _heuristic_analysis(
        self,
        query: str,
        documents: List[Document],
        quality_score: float,
        concepts_covered: List[str],
        concepts_missing: List[str],
    ) -> GapAnalysis:
        """Fallback heuristic-based gap analysis."""
        gaps = []
        
        # Gap 1: Low quality score
        if quality_score < 0.5:
            gaps.append(InformationGap(
                gap_type=GapType.INCOMPLETE_COVERAGE,
                description="Retrieved documents have low relevance scores",
                severity=0.7,
                suggested_action="expand_query",
                suggested_query=None,
            ))
        
        # Gap 2: Missing concepts
        if concepts_missing:
            for concept in concepts_missing[:3]:  # Top 3 missing concepts
                gaps.append(InformationGap(
                    gap_type=GapType.MISSING_TOPIC,
                    description=f"Missing information about: {concept}",
                    severity=0.6,
                    suggested_action="keyword_search",
                    suggested_query=f"{query} {concept}",
                ))
        
        # Gap 3: Too few results
        if len(documents) < 3:
            gaps.append(InformationGap(
                gap_type=GapType.INSUFFICIENT_CONTEXT,
                description="Too few documents retrieved",
                severity=0.5,
                suggested_action="relax_filters",
                suggested_query=None,
            ))
        
        # Determine primary gap
        primary_gap = None
        if gaps:
            # Sort by severity and take the first
            gaps.sort(key=lambda g: g.severity, reverse=True)
            primary_gap = gaps[0].description
        
        # Build feedback
        feedback = ""
        if gaps:
            feedback = f"Consider: {gaps[0].suggested_action}. {gaps[0].description}"
        
        return GapAnalysis(
            has_gaps=len(gaps) > 0,
            gaps=gaps,
            coverage_score=quality_score,
            primary_gap=primary_gap,
            feedback_for_retry=feedback,
        )


# ============================================================================
# Module-level convenience function
# ============================================================================

@traceable(name="identify_gaps", run_type="tool")
def identify_gaps(
    query: str,
    documents: List[Document],
    quality_score: float = 0.5,
    concepts_covered: Optional[List[str]] = None,
    concepts_missing: Optional[List[str]] = None,
    use_llm: bool = True,
) -> GapAnalysis:
    """
    Identify information gaps in retrieved documents.
    
    Convenience function that creates and invokes GapIdentificationTool.
    
    Args:
        query: The search query
        documents: Retrieved documents to analyze
        quality_score: Overall quality score from assessment
        concepts_covered: Concepts found in results
        concepts_missing: Concepts not found
        use_llm: Whether to use LLM (True) or heuristics (False)
        
    Returns:
        GapAnalysis with identified gaps and suggestions
    """
    tool = GapIdentificationTool(use_llm=use_llm)
    return tool(query, documents, quality_score, concepts_covered, concepts_missing)
