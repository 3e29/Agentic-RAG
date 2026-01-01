"""
LangGraph Workflow for Hadith RAG System

This module defines the complete LangGraph workflow that orchestrates:
1. Query Analysis Agent - Parse and understand the query
2. Retrieval Agent - Execute search strategies
3. Evaluation Agent - Assess quality and decide continue/stop

**Architecture:**
- Single-pass agents with workflow-controlled loops
- Conditional routing based on evaluation decisions
- Max iteration limits to prevent infinite loops

**Key Design Decisions:**
- Agents are stateless single-pass functions
- State is passed through AgentState TypedDict
- Loop control is in the workflow, not in agents

Production Standards:
- LangGraph StateGraph for orchestration
- Clean separation of concerns
- Observable via LangSmith tracing
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState

# Import agents (Nodes)
from src.agents.query_analysis import query_analysis_agent
from src.agents.retrieval import retrieval_agent
from src.agents.evaluation import evaluation_agent

# Configuration
MAX_WORKFLOW_ITERATIONS = 3  # Maximum retrieval-evaluation cycles


def create_workflow() -> StateGraph:
    """
    Create and compile the Hadith RAG workflow.
    
    Flow:
    START -> query_analysis -> retrieval -> evaluation -> (continue/stop)
                                    ^                          |
                                    |__________________________| (if continue)
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # 1. Initialize the Graph
    builder = StateGraph(AgentState)

    # 2. Add Nodes
    builder.add_node("query_analysis", query_analysis_agent)
    builder.add_node("retrieval", retrieval_agent)
    builder.add_node("evaluation", evaluation_agent)

    # 3. Define Flow (Edges)
    # Start -> Query Analysis
    builder.add_edge(START, "query_analysis")
    
    # Query Analysis -> Retrieval
    builder.add_edge("query_analysis", "retrieval")

    # Retrieval -> Evaluation
    builder.add_edge("retrieval", "evaluation")

    # 4. Define Conditional Edges (The "Loop" Logic)
    # This is where the Evaluation Agent's decision controls the flow
    builder.add_conditional_edges(
        "evaluation",
        route_after_evaluation,
        {
            "continue": "retrieval",  # Loop back for another search pass
            "stop": END               # Proceed to output (or generation agent)
        }
    )

    # 5. Compile and return
    return builder.compile()


def route_after_evaluation(state: AgentState) -> Literal["continue", "stop"]:
    """
    Routing function that decides the next step based on Evaluation Agent output.
    
    Decision Logic:
    1. If evaluation status is STOP -> stop
    2. If max iterations reached -> stop (safety limit)
    3. If evaluation status is CONTINUE and iterations < max -> continue
    
    Args:
        state: Current AgentState with evaluation results
        
    Returns:
        "continue" to loop back to retrieval, "stop" to end
    """
    # Get evaluation metadata
    metadata = state.get("metadata", {}) or {}
    evaluation = metadata.get("evaluation", {})
    
    # Get the evaluation decision
    status = evaluation.get("status", "STOP")
    iteration = evaluation.get("iteration", 1)
    max_iterations_reached = evaluation.get("max_iterations_reached", False)
    
    # Safety check: prevent infinite loops
    if iteration >= MAX_WORKFLOW_ITERATIONS:
        return "stop"
    
    if max_iterations_reached:
        return "stop"
    
    # Route based on evaluation decision
    if status == "CONTINUE":
        return "continue"
    
    return "stop"


# ============================================================================
# Convenience Functions
# ============================================================================

def run_workflow(query: str) -> AgentState:
    """
    Run the complete workflow for a query.
    
    Convenience function for testing and standalone use.
    
    Args:
        query: The user's search query
        
    Returns:
        Final AgentState with all results
    """
    workflow = create_workflow()
    
    # Initialize state
    initial_state: AgentState = {
        "original_query": query,
        "normalized_query": None,
        "corrected_query": None,
        "input_source": None,
        "query_intent": None,
        "target_collections": None,
        "sub_queries": None,
        "retrieved_docs": None,
        "evaluation_feedback": None,
        "confidence_score": None,
        "missing_information_gaps": None,
        "language": None,
        "desired_output_language": None,
        "metadata": {},
    }
    
    # Execute workflow
    result = workflow.invoke(initial_state)
    
    return result


# ============================================================================
# Module exports
# ============================================================================

# Create a default workflow instance for import
default_workflow = create_workflow()


if __name__ == "__main__":
    # Quick test
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_query = "أحاديث عن الصبر"
    print(f"Testing workflow with: {test_query}")
    
    result = run_workflow(test_query)
    
    print(f"\nResults:")
    print(f"- Query Intent: {result.get('query_intent')}")
    print(f"- Docs Retrieved: {len(result.get('retrieved_docs', []))}")
    print(f"- Confidence Score: {result.get('confidence_score')}")
    print(f"- Evaluation Feedback: {result.get('evaluation_feedback')}")