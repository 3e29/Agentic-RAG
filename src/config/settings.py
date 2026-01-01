"""
Application Configuration Settings

This module centralizes all configuration constants for the Hadith RAG system.
These values can be overridden via environment variables for different environments
(development, staging, production) without requiring code changes.

Configuration Categories:
- Agent Behavior: ReAct loop iterations, retry logic
- Retrieval Parameters: Top-k values, search thresholds
- Performance Tuning: Parallel search parameters

Usage:
    from src.config.settings import MAX_AGENT_ITERATIONS, DEFAULT_TOP_K
    
    for iteration in range(MAX_AGENT_ITERATIONS):
        # Agent logic here
        pass
"""

import os

# =============================================================================
# AGENT BEHAVIOR CONFIGURATION
# =============================================================================

# Maximum number of ReAct loop iterations for autonomous retrieval agent
# Higher values allow more iterative refinement but increase latency and cost
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))

# Maximum number of retry attempts for failed operations
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

# Number of final results to return after reranking
# This is the top-k value presented to the user
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# Number of candidates to fetch during initial retrieval for cross-encoder reranking
# Higher values improve reranking quality but increase processing time
# Recommended: 10x the final top-k value
PARALLEL_SEARCH_K = int(os.getenv("PARALLEL_SEARCH_K", "50"))
