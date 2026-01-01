"""
Evaluation Tools for Hadith RAG System

This module provides tools for evaluating retrieval quality and identifying
gaps in search results. These tools support the Evaluation Agent in deciding
whether to continue searching or finalize results.

Tools:
- QualityAssessmentTool: Assesses if retrieved docs answer the query
- GapIdentificationTool: Identifies missing information in results
- GroundingValidator: Validates docs contain requested concepts

Production Standards:
- Pydantic V2 for input/output validation
- LangSmith tracing for observability
- Graceful fallbacks for LLM failures
"""

from src.tools.evaluation.schemas import (
    EvaluationStatus,
    QualityAssessment,
    GapAnalysis,
    GroundingResult,
    EvaluationResult,
)
from src.tools.evaluation.quality_assessment import (
    QualityAssessmentTool,
    assess_quality,
)
from src.tools.evaluation.gap_identification import (
    GapIdentificationTool,
    identify_gaps,
)
from src.tools.evaluation.grounding_validator import (
    GroundingValidatorTool,
    validate_grounding,
)

__all__ = [
    # Enums
    "EvaluationStatus",
    # Schemas
    "QualityAssessment",
    "GapAnalysis",
    "GroundingResult",
    "EvaluationResult",
    # Tools
    "QualityAssessmentTool",
    "assess_quality",
    "GapIdentificationTool",
    "identify_gaps",
    "GroundingValidatorTool",
    "validate_grounding",
]
