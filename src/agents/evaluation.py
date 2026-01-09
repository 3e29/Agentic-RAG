"""
Evaluation Agent for Hadith RAG System

This agent evaluates retrieval quality and decides whether to:
1. CONTINUE searching (with feedback for improvement)
2. STOP and proceed to synthesis

**Separation of Concerns:**
- Retrieval Agent: "I found these docs"
- Evaluation Agent: "These are good/bad, here's why, here's what to do next"

**Architecture:**
- Uses three evaluation tools: QualityAssessment, GapIdentification, GroundingValidator
- Combines tool outputs into a unified decision
- Provides actionable feedback for retry strategies

**LangGraph Integration:**
- Node function: evaluation_agent(state) -> dict
- Reads: retrieved_docs, query from state
- Writes: evaluation feedback, confidence, decision to state

Production Standards:
- LangSmith tracing for observability
- Pydantic V2 for type safety
- Graceful degradation on tool failures
"""

import logging
import time
from typing import Dict, Any, List, Optional

from langsmith import traceable

from src.graph.state import AgentState
from src.tools.evaluation.schemas import (
    EvaluationStatus,
    EvaluationResult,
    QualityAssessment,
    GapAnalysis,
    GroundingResult,
)
from src.tools.evaluation.quality_assessment import QualityAssessmentTool
from src.tools.evaluation.gap_identification import GapIdentificationTool
from src.tools.evaluation.grounding_validator import GroundingValidatorTool
from src.tools.retrieval.schemas import Document

# Configuration
from src.config.settings import (
    MAX_AGENT_ITERATIONS,
    DEFAULT_TOP_K,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================

# Evaluation thresholds
MIN_QUALITY_SCORE_TO_STOP = 0.65      # Minimum quality to stop searching
MIN_GROUNDING_SCORE_TO_STOP = 0.5     # Minimum grounding to stop
MAX_EVAL_ITERATIONS = 2               # Max evaluation cycles before force-stop
MIN_DOCS_FOR_EVALUATION = 1           # Minimum docs needed to evaluate

# Weights for combined score
QUALITY_WEIGHT = 0.5
GROUNDING_WEIGHT = 0.3
COVERAGE_WEIGHT = 0.2


# ============================================================================
# Evaluation Agent Node
# ============================================================================

@traceable(name="evaluation_agent", run_type="chain")
def evaluation_agent(state: AgentState, **kwargs) -> Dict[str, Any]:
    """
    Evaluation Agent node for LangGraph workflow.
    
    Evaluates the quality of retrieved documents and decides whether
    to continue searching or proceed to synthesis.
    
    Args:
        state: Current AgentState with retrieved_docs and query
        **kwargs: Additional arguments from LangGraph (e.g., config)
        
    Returns:
        Dictionary with evaluation results and updated metadata:
        - evaluation_feedback: Actionable feedback string
        - confidence_score: Float confidence in results
        - missing_information_gaps: List of identified gaps
        - metadata.evaluation: Full evaluation details
    """
    start_time = time.time()
    
    # Extract inputs from state
    # For evaluation, use ORIGINAL query to check if all parts are covered
    # (especially for compound queries that were decomposed)
    original_query = state.get("original_query", "")
    corrected_query = (
        state.get("corrected_query") or
        state.get("normalized_query") or
        original_query
    )
    
    # Use original query for comprehensive evaluation of compound queries
    # This ensures we check if BOTH parts of "X و Y" are covered
    query_for_evaluation = original_query or corrected_query
    
    retrieved_docs = state.get("retrieved_docs", [])
    metadata = state.get("metadata", {}) or {}
    
    # Track sub-queries for better evaluation context
    sub_queries = state.get("sub_queries", [])
    search_sub_queries = state.get("search_sub_queries", [])
    
    # Get current iteration count
    eval_metadata = metadata.get("evaluation", {})
    current_iteration = eval_metadata.get("iteration", 0) + 1
    
    logger.info(
        f"Evaluation Agent starting (iteration {current_iteration}): "
        f"query='{query_for_evaluation[:50]}...', docs={len(retrieved_docs)}, "
        f"sub_queries={len(sub_queries) if sub_queries else 0}"
    )
    
    # Convert dicts to Document objects if needed
    documents = _ensure_documents(retrieved_docs)
    
    # Run evaluation pipeline with original query for comprehensive coverage check
    evaluation_result = _evaluate_results(
        query=query_for_evaluation,
        documents=documents,
        current_iteration=current_iteration,
        max_iterations=MAX_EVAL_ITERATIONS,
    )
    
    # Build response
    execution_time = (time.time() - start_time) * 1000
    
    # Update metadata - only include evaluation-specific data
    # (LangGraph will merge this with existing state metadata)
    evaluation_metadata = {
        "evaluation": {
            "status": evaluation_result.status.value,
            "confidence_score": evaluation_result.confidence_score,
            "reasoning": evaluation_result.reasoning,
            "iteration": current_iteration,
            "max_iterations_reached": evaluation_result.max_iterations_reached,
            "execution_time_ms": execution_time,
            "quality_score": (
                evaluation_result.quality_assessment.overall_score
                if evaluation_result.quality_assessment else None
            ),
            "grounding_score": (
                evaluation_result.grounding_result.grounding_score
                if evaluation_result.grounding_result else None
            ),
            "coverage_score": (
                evaluation_result.gap_analysis.coverage_score
                if evaluation_result.gap_analysis else None
            ),
            "suggested_actions": evaluation_result.suggested_actions,
        }
    }
    
    # Extract gaps as list of strings
    gaps_list = []
    if evaluation_result.gap_analysis and evaluation_result.gap_analysis.gaps:
        gaps_list = [gap.description for gap in evaluation_result.gap_analysis.gaps]
    
    logger.info(
        f"Evaluation complete: status={evaluation_result.status.value}, "
        f"confidence={evaluation_result.confidence_score:.2f}, "
        f"time={execution_time:.1f}ms"
    )
    
    # Merge evaluation metadata with existing state metadata
    existing_metadata = state.get("metadata", {}) or {}
    merged_metadata = {**existing_metadata, **evaluation_metadata}
    
    return {
        "evaluation_feedback": evaluation_result.feedback,
        "confidence_score": evaluation_result.confidence_score,
        "missing_information_gaps": gaps_list,
        "metadata": merged_metadata,  # Merged metadata preserving retrieval info
    }


# ============================================================================
# Core Evaluation Logic
# ============================================================================

@traceable(name="evaluate_results", run_type="chain")
def _evaluate_results(
    query: str,
    documents: List[Document],
    current_iteration: int,
    max_iterations: int,
) -> EvaluationResult:
    """
    Run the full evaluation pipeline.
    
    Combines quality assessment, gap identification, and grounding validation
    into a unified evaluation decision.
    
    Args:
        query: The search query
        documents: Retrieved documents to evaluate
        current_iteration: Current evaluation iteration
        max_iterations: Maximum allowed iterations
        
    Returns:
        EvaluationResult with decision and feedback
    """
    # Check for max iterations
    if current_iteration >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached, forcing STOP")
        return EvaluationResult(
            status=EvaluationStatus.STOP,
            confidence_score=0.5,  # Uncertain but stopping
            reasoning=f"Maximum evaluation iterations ({max_iterations}) reached",
            feedback="Search completed due to iteration limit",
            suggested_actions=[],
            iteration=current_iteration,
            max_iterations_reached=True,
        )
    
    # Check for minimum documents
    if len(documents) < MIN_DOCS_FOR_EVALUATION:
        logger.info(f"Insufficient documents ({len(documents)}), suggesting CONTINUE")
        return EvaluationResult(
            status=EvaluationStatus.CONTINUE,
            confidence_score=0.1,
            reasoning=f"Only {len(documents)} documents retrieved, need more",
            feedback="Expand search query or relax filters",
            suggested_actions=["expand_query", "relax_filters"],
            iteration=current_iteration,
            max_iterations_reached=False,
        )
    
    # Step 1: Quality Assessment
    quality_result = _run_quality_assessment(query, documents)
    
    # Step 2: Gap Identification
    gap_result = _run_gap_identification(
        query=query,
        documents=documents,
        quality_score=quality_result.overall_score,
        concepts_covered=quality_result.concepts_covered,
        concepts_missing=quality_result.concepts_missing,
    )
    
    # Step 3: Grounding Validation
    grounding_result = _run_grounding_validation(
        query=query,
        documents=documents,
        key_concepts=quality_result.query_concepts,
    )
    
    # Step 4: Combine into decision
    return _make_decision(
        quality_result=quality_result,
        gap_result=gap_result,
        grounding_result=grounding_result,
        current_iteration=current_iteration,
    )


@traceable(name="quality_assessment", run_type="tool")
def _run_quality_assessment(
    query: str,
    documents: List[Document],
) -> QualityAssessment:
    """Run quality assessment tool with error handling."""
    try:
        tool = QualityAssessmentTool(use_llm=True)
        return tool(query, documents)
    except Exception as e:
        logger.warning(f"Quality assessment failed: {e}, using fallback")
        # Fallback: use heuristic
        tool = QualityAssessmentTool(use_llm=False)
        return tool(query, documents)


@traceable(name="gap_identification", run_type="tool")
def _run_gap_identification(
    query: str,
    documents: List[Document],
    quality_score: float,
    concepts_covered: List[str],
    concepts_missing: List[str],
) -> GapAnalysis:
    """Run gap identification tool with error handling."""
    try:
        tool = GapIdentificationTool(use_llm=True)
        return tool(
            query=query,
            documents=documents,
            quality_score=quality_score,
            concepts_covered=concepts_covered,
            concepts_missing=concepts_missing,
        )
    except Exception as e:
        logger.warning(f"Gap identification failed: {e}, using fallback")
        tool = GapIdentificationTool(use_llm=False)
        return tool(query, documents, quality_score, concepts_covered, concepts_missing)


@traceable(name="grounding_validation", run_type="tool")
def _run_grounding_validation(
    query: str,
    documents: List[Document],
    key_concepts: Optional[List[str]],
) -> GroundingResult:
    """Run grounding validation tool with error handling."""
    try:
        tool = GroundingValidatorTool()
        return tool(query, documents, key_concepts)
    except Exception as e:
        logger.warning(f"Grounding validation failed: {e}, using default")
        return GroundingResult(
            is_grounded=True,  # Assume grounded on failure
            grounding_score=0.7,
            concept_groundings=[],
            ungrounded_concepts=[],
            hallucination_risk="medium",
        )


def _make_decision(
    quality_result: QualityAssessment,
    gap_result: GapAnalysis,
    grounding_result: GroundingResult,
    current_iteration: int,
) -> EvaluationResult:
    """
    Combine tool results into a final decision.
    
    Decision logic:
    1. If quality >= threshold AND grounding >= threshold -> STOP
    2. If critical gaps exist AND iterations < max -> CONTINUE
    3. Otherwise -> STOP (with lower confidence)
    """
    # Calculate combined score
    combined_score = (
        QUALITY_WEIGHT * quality_result.overall_score +
        GROUNDING_WEIGHT * grounding_result.grounding_score +
        COVERAGE_WEIGHT * gap_result.coverage_score
    )
    
    # Determine status
    should_stop = (
        quality_result.overall_score >= MIN_QUALITY_SCORE_TO_STOP and
        grounding_result.grounding_score >= MIN_GROUNDING_SCORE_TO_STOP and
        not gap_result.has_gaps
    )
    
    # Check for critical gaps that warrant retry
    critical_gaps = [
        gap for gap in gap_result.gaps
        if gap.severity >= 0.7
    ]
    
    if should_stop:
        status = EvaluationStatus.STOP
        reasoning = (
            f"Results sufficient: quality={quality_result.overall_score:.2f}, "
            f"grounding={grounding_result.grounding_score:.2f}"
        )
        feedback = "Results are ready for synthesis"
        suggested_actions = []
        
    elif critical_gaps and current_iteration < MAX_EVAL_ITERATIONS:
        status = EvaluationStatus.CONTINUE
        reasoning = f"Critical gaps found: {critical_gaps[0].description}"
        feedback = gap_result.feedback_for_retry or "Retry with expanded query"
        suggested_actions = [gap.suggested_action for gap in critical_gaps[:3]]
        
    elif quality_result.overall_score < 0.3:
        # Very low quality - definitely retry
        status = EvaluationStatus.CONTINUE
        reasoning = f"Quality too low: {quality_result.overall_score:.2f}"
        feedback = "Results are not relevant, try different search strategy"
        suggested_actions = ["expand_query", "relax_filters", "keyword_search"]
        
    else:
        # Marginal results - stop but with lower confidence
        status = EvaluationStatus.STOP
        reasoning = (
            f"Marginal results: quality={quality_result.overall_score:.2f}, "
            f"proceeding with available data"
        )
        feedback = "Results may be incomplete but proceeding"
        suggested_actions = []
    
    return EvaluationResult(
        status=status,
        confidence_score=combined_score,
        reasoning=reasoning,
        quality_assessment=quality_result,
        gap_analysis=gap_result,
        grounding_result=grounding_result,
        feedback=feedback,
        suggested_actions=suggested_actions,
        iteration=current_iteration,
        max_iterations_reached=False,
    )


# ============================================================================
# Utility Functions
# ============================================================================

def _ensure_documents(docs: List[Any]) -> List[Document]:
    """Convert dict representations to Document objects if needed."""
    result = []
    for doc in docs:
        if isinstance(doc, Document):
            result.append(doc)
        elif isinstance(doc, dict):
            try:
                result.append(Document(**doc))
            except Exception as e:
                logger.warning(f"Could not convert dict to Document: {e}")
        else:
            # Try to use as-is if it has required attributes
            if hasattr(doc, 'chunk_id') and hasattr(doc, 'text'):
                result.append(doc)
    return result


# ============================================================================
# Alternative: Class-based Agent (for dependency injection)
# ============================================================================

class EvaluationAgent:
    """
    Class-based Evaluation Agent for more control over configuration.
    
    Use this when you need to customize thresholds or inject dependencies.
    """
    
    def __init__(
        self,
        min_quality_threshold: float = MIN_QUALITY_SCORE_TO_STOP,
        min_grounding_threshold: float = MIN_GROUNDING_SCORE_TO_STOP,
        max_iterations: int = MAX_EVAL_ITERATIONS,
        use_llm: bool = True,
    ):
        """
        Initialize the evaluation agent.
        
        Args:
            min_quality_threshold: Minimum quality to stop
            min_grounding_threshold: Minimum grounding to stop
            max_iterations: Max evaluation iterations
            use_llm: Whether to use LLM for assessments
        """
        self.min_quality = min_quality_threshold
        self.min_grounding = min_grounding_threshold
        self.max_iterations = max_iterations
        self.use_llm = use_llm
        
        # Initialize tools
        self.quality_tool = QualityAssessmentTool(use_llm=use_llm)
        self.gap_tool = GapIdentificationTool(use_llm=use_llm)
        self.grounding_tool = GroundingValidatorTool()
    
    @traceable(name="evaluation_agent_class", run_type="chain")
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute evaluation on the current state.
        
        Same interface as the functional evaluation_agent.
        """
        return evaluation_agent(state)
    
    def evaluate(
        self,
        query: str,
        documents: List[Document],
        iteration: int = 1,
    ) -> EvaluationResult:
        """
        Direct evaluation method for testing/standalone use.
        
        Args:
            query: The search query
            documents: Documents to evaluate
            iteration: Current iteration number
            
        Returns:
            EvaluationResult with decision and feedback
        """
        return _evaluate_results(
            query=query,
            documents=documents,
            current_iteration=iteration,
            max_iterations=self.max_iterations,
        )
