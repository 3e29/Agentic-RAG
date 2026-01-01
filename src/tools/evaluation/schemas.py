"""
Pydantic V2 Schemas for Evaluation Tools

This module defines all data contracts for the evaluation system.
These schemas ensure type safety and validation across evaluation operations.

Production Standards:
- Strict validation with Pydantic V2
- Field descriptions for documentation
- Default values for optional fields
- Serialization-ready for API responses
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Enums for Type Safety
# ============================================================================

class EvaluationStatus(str, Enum):
    """
    Decision status from the Evaluation Agent.
    
    - CONTINUE: Results are insufficient, retry with feedback
    - STOP: Results are sufficient, proceed to synthesis
    """
    CONTINUE = "CONTINUE"
    STOP = "STOP"


class RelevanceLevel(str, Enum):
    """Relevance level for individual documents."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"


class GapType(str, Enum):
    """Type of information gap identified."""
    MISSING_TOPIC = "missing_topic"          # Key topic not covered
    INCOMPLETE_COVERAGE = "incomplete"       # Topic partially covered
    WRONG_COLLECTION = "wrong_collection"    # Results from wrong source
    WRONG_LANGUAGE = "wrong_language"        # Results in wrong language
    INSUFFICIENT_CONTEXT = "insufficient"    # Not enough supporting info


# ============================================================================
# Quality Assessment Schemas
# ============================================================================

class DocumentRelevance(BaseModel):
    """Relevance assessment for a single document."""
    model_config = ConfigDict(extra="forbid")
    
    chunk_id: str = Field(description="ID of the evaluated document")
    relevance: RelevanceLevel = Field(description="Assessed relevance level")
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="Numeric relevance score (0-1)"
    )
    reason: str = Field(description="Explanation for the relevance rating")
    key_concepts_matched: List[str] = Field(
        default_factory=list,
        description="Query concepts found in this document"
    )


class QualityAssessment(BaseModel):
    """
    Output from QualityAssessmentTool.
    
    Assesses whether the retrieved documents collectively answer the query.
    """
    model_config = ConfigDict(extra="forbid")
    
    overall_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall quality score (0-1)"
    )
    is_sufficient: bool = Field(
        description="Whether results sufficiently answer the query"
    )
    reasoning: str = Field(
        description="Explanation for the assessment"
    )
    document_assessments: List[DocumentRelevance] = Field(
        default_factory=list,
        description="Per-document relevance assessments"
    )
    query_concepts: List[str] = Field(
        default_factory=list,
        description="Key concepts extracted from the query"
    )
    concepts_covered: List[str] = Field(
        default_factory=list,
        description="Query concepts found in results"
    )
    concepts_missing: List[str] = Field(
        default_factory=list,
        description="Query concepts NOT found in results"
    )


# ============================================================================
# Gap Identification Schemas
# ============================================================================

class InformationGap(BaseModel):
    """A specific gap identified in the search results."""
    model_config = ConfigDict(extra="forbid")
    
    gap_type: GapType = Field(description="Type of information gap")
    description: str = Field(description="Description of what is missing")
    severity: float = Field(
        ge=0.0, le=1.0,
        description="Severity of the gap (0=minor, 1=critical)"
    )
    suggested_action: str = Field(
        description="Suggested action to fill the gap"
    )
    suggested_query: Optional[str] = Field(
        default=None,
        description="Alternative query that might fill the gap"
    )


class GapAnalysis(BaseModel):
    """
    Output from GapIdentificationTool.
    
    Identifies what information is missing from the current results.
    """
    model_config = ConfigDict(extra="forbid")
    
    has_gaps: bool = Field(
        description="Whether significant gaps were found"
    )
    gaps: List[InformationGap] = Field(
        default_factory=list,
        description="List of identified gaps"
    )
    coverage_score: float = Field(
        ge=0.0, le=1.0,
        description="How well results cover the query (0-1)"
    )
    primary_gap: Optional[str] = Field(
        default=None,
        description="The most critical missing information"
    )
    feedback_for_retry: str = Field(
        default="",
        description="Actionable feedback if retry is needed"
    )


# ============================================================================
# Grounding Validation Schemas
# ============================================================================

class ConceptGrounding(BaseModel):
    """Grounding check for a single concept."""
    model_config = ConfigDict(extra="forbid")
    
    concept: str = Field(description="The concept being checked")
    is_grounded: bool = Field(
        description="Whether concept appears in documents"
    )
    evidence_chunks: List[str] = Field(
        default_factory=list,
        description="Chunk IDs where concept was found"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the grounding check"
    )


class GroundingResult(BaseModel):
    """
    Output from GroundingValidatorTool.
    
    Validates that documents actually contain the requested concepts.
    """
    model_config = ConfigDict(extra="forbid")
    
    is_grounded: bool = Field(
        description="Whether all key concepts are grounded"
    )
    grounding_score: float = Field(
        ge=0.0, le=1.0,
        description="Proportion of concepts that are grounded"
    )
    concept_groundings: List[ConceptGrounding] = Field(
        default_factory=list,
        description="Per-concept grounding checks"
    )
    ungrounded_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts NOT found in any document"
    )
    hallucination_risk: str = Field(
        default="low",
        description="Risk level for hallucination (low/medium/high)"
    )


# ============================================================================
# Combined Evaluation Result
# ============================================================================

class EvaluationResult(BaseModel):
    """
    Combined output from the Evaluation Agent.
    
    Aggregates quality, gaps, and grounding into a single decision.
    """
    model_config = ConfigDict(extra="forbid")
    
    # Decision
    status: EvaluationStatus = Field(
        description="CONTINUE or STOP decision"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the evaluation decision"
    )
    reasoning: str = Field(
        description="Explanation for the decision"
    )
    
    # Component Results
    quality_assessment: Optional[QualityAssessment] = Field(
        default=None,
        description="Quality assessment results"
    )
    gap_analysis: Optional[GapAnalysis] = Field(
        default=None,
        description="Gap identification results"
    )
    grounding_result: Optional[GroundingResult] = Field(
        default=None,
        description="Grounding validation results"
    )
    
    # Feedback for Retry
    feedback: str = Field(
        default="",
        description="Actionable feedback for retry"
    )
    suggested_actions: List[str] = Field(
        default_factory=list,
        description="Specific actions to improve results"
    )
    
    # Iteration Tracking
    iteration: int = Field(
        default=1,
        description="Current evaluation iteration"
    )
    max_iterations_reached: bool = Field(
        default=False,
        description="Whether max iterations have been reached"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return self.model_dump()
