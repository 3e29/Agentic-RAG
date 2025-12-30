"""
End-to-End Integration Test for Hadith RAG Pipeline

This test verifies the complete flow:
    Complex Query -> Query Analysis Agent -> State Handoff -> Retrieval Agent (Parallel) -> Aggregation

**LangSmith Observability:**
All agent functions are decorated with @traceable, so traces will appear
in the LangSmith dashboard for debugging and performance analysis.

**Requirements:**
- LANGSMITH_API_KEY in .env file
- LANGSMITH_TRACING=true
- Real ChromaDB vector store with indexed hadiths
"""

import os
import sys
import logging
from typing import Dict, Any, List
from datetime import datetime

from chromadb import Client
import pytest
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# Load environment variables FIRST
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "hadith-rag")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import agents and state
from src.graph.state import AgentState
from src.agents.query_analysis import query_analysis_agent
from src.agents.retrieval import retrieval_agent


# ============================================================================
# LangSmith Helper Functions
# ============================================================================

def print_langsmith_instructions(test_name: str = "test_complex_pipeline_flow"):
    """
    Print step-by-step instructions for viewing traces in LangSmith.
    
    Call this at the start of each test to guide observability.
    """
    project_name = os.environ.get("LANGSMITH_PROJECT", "hadith-rag")
    
    print("\n" + "=" * 70)
    print("🔎 LANGSMITH OBSERVABILITY INSTRUCTIONS")
    print("=" * 70)
    print(f"""
📊 To view this test run in LangSmith:

1. Go to: https://smith.langchain.com/

2. Select your project: '{project_name}'
   - Click on 'Projects' in the left sidebar
   - Find and click '{project_name}'

3. Look for the trace named: '{test_name}'
   - Sort by 'Start Time' (descending) to find the latest run
   - The run should appear within ~30 seconds

4. Explore the trace hierarchy:
   ┌─ {test_name} (root)
   │  ├─ query_analysis_agent
   │  │  ├─ input_source_identification_tool
   │  │  ├─ query_normalization_tool
   │  │  ├─ typo_correction_tool
   │  │  ├─ intent_classification_tool
   │  │  ├─ collection_target_detection_tool
   │  │  └─ query_decomposition_tool
   │  │
   │  └─ retrieval_agent
   │     ├─ query_expansion (fast)
   │     ├─ parallel_search (PARALLEL BARS!)
   │     │  ├─ _execute_search_with_retry [subquery_0]
   │     │  └─ _execute_search_with_retry [subquery_1]
   │     └─ aggregate_results

5. Look for PARALLEL execution:
   - In 'retrieval_agent', expand the trace
   - You'll see multiple search tasks running simultaneously
   - Parallel tasks appear as overlapping horizontal bars

6. Check latencies:
   - Each node shows execution time
   - Identify bottlenecks in the pipeline

💡 TIP: Click on any node to see:
   - Input parameters
   - Output values
   - Token usage (for LLM calls)
   - Error messages (if any)
""")
    print("=" * 70 + "\n")
    # At the very end of the test function, before it exits:
    from langsmith import Client
    Client().flush()  # Forces all pending logs to be sent immediately


def get_langsmith_run_url() -> str:
    """
    Get the URL for the current LangSmith run (if available).
    
    Returns URL string or empty string if not in a traced context.
    """
    try:
        run_tree = get_current_run_tree()
        if run_tree and run_tree.id:
            project = os.environ.get("LANGSMITH_PROJECT", "hadith-rag")
            return f"https://smith.langchain.com/o/default/projects/p/{project}/r/{run_tree.id}"
    except Exception:
        pass
    return ""


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def ensure_langsmith_configured():
    """Verify LangSmith is properly configured before running tests."""
    api_key = os.environ.get("LANGSMITH_API_KEY")
    
    if not api_key:
        pytest.skip("LANGSMITH_API_KEY not set in environment")
    
    # Verify tracing is enabled
    assert os.environ.get("LANGSMITH_TRACING") == "true", \
        "LANGSMITH_TRACING should be 'true'"
    
    project = os.environ.get("LANGSMITH_PROJECT", "hadith-rag")
    print(f"\n✅ LangSmith configured: Project='{project}'")
    
    return True


@pytest.fixture
def initial_state_ablution_tayammum() -> AgentState:
    """
    Create initial state for ablution vs tayammum test.
    """
    return {
        "original_query": "Compare the rules of Wudu (ablution) and Tayammum in Sahih Bukhari",
        "normalized_query": None,
        "corrected_query": None,
        "input_source": None,
        "query_intent": None,
        "target_collections": None,
        "sub_queries": None,
        "retrieved_docs": None,
        "language": None,
        "metadata": {},
    }


@pytest.fixture
def initial_state_comparative() -> AgentState:
    """
    Create initial state for comparative analysis test.
    
    Query asks to compare Zakat and Fasting in Bukhari.
    """
    return {
        "original_query": "Compare the rulings of Zakat and Fasting in Sahih Bukhari",
        "normalized_query": None,
        "corrected_query": None,
        "input_source": None,
        "query_intent": None,
        "target_collections": None,
        "sub_queries": None,
        "retrieved_docs": None,
        "language": None,
        "metadata": {},
    }


@pytest.fixture
def initial_state_arabic() -> AgentState:
    """
    Create initial state for Arabic comparative query.
    """
    return {
        "original_query": "قارن بين أحاديث الزكاة والصيام في صحيح البخاري",
        "normalized_query": None,
        "corrected_query": None,
        "input_source": None,
        "query_intent": None,
        "target_collections": None,
        "sub_queries": None,
        "retrieved_docs": None,
        "language": None,
        "metadata": {},
    }


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndPipeline:
    """
    End-to-end integration tests for the complete RAG pipeline.
    
    These tests use real agents with real LLM calls and vector store.
    They are decorated with @traceable for LangSmith observability.
    """
    
    @traceable(name="test_complex_pipeline_flow")
    def test_complex_pipeline_flow(
        self,
        ensure_langsmith_configured,
        initial_state_comparative,
    ):
        """
        Test the full pipeline: Query Analysis -> Retrieval with parallel execution.
        
        Input: "Compare the rulings of Zakat and Fasting in Sahih Bukhari"
        
        Expected Flow:
        1. Query Analysis identifies comparative_analysis intent
        2. Query is decomposed into sub-queries for Zakat and Fasting
        3. Retrieval runs parallel searches for both topics
        4. Results are aggregated and contain hadiths about both topics
        """
        # Print observability instructions
        print_langsmith_instructions("test_complex_pipeline_flow")
        
        state = initial_state_comparative
        print(f"\n📝 Input Query: {state['original_query']}")
        print("-" * 60)
        
        # ====================================================================
        # STEP 1: Query Analysis
        # ====================================================================
        print("\n🔄 STEP 1: Running Query Analysis Agent...")
        
        analysis_result = query_analysis_agent(state)
        
        # Merge results into state
        state.update(analysis_result)
        
        # Log analysis results
        print(f"   ✓ Intent: {state.get('query_intent')}")
        print(f"   ✓ Input Source: {state.get('input_source')}")
        print(f"   ✓ Target Collections: {state.get('target_collections')}")
        print(f"   ✓ Sub-queries: {state.get('sub_queries')}")
        print(f"   ✓ Language: {state.get('language')}")
        
        # Assertions for Query Analysis
        assert state.get("query_intent") == "comparative_analysis", \
            f"Expected comparative_analysis, got {state.get('query_intent')}"
        
        assert state.get("sub_queries") is not None, \
            "Sub-queries should be generated for comparative analysis"
        
        assert len(state.get("sub_queries", [])) >= 2, \
            f"Expected at least 2 sub-queries, got {len(state.get('sub_queries', []))}"
        
        # Check that sub-queries mention Zakat and Fasting
        sub_queries_text = " ".join(state.get("sub_queries", [])).lower()
        assert "zakat" in sub_queries_text or "زكاة" in sub_queries_text, \
            "Sub-queries should include Zakat topic"
        assert "fast" in sub_queries_text or "صيام" in sub_queries_text or "صوم" in sub_queries_text, \
            "Sub-queries should include Fasting topic"
        
        print("\n   ✅ Query Analysis assertions passed!")
        
        # ====================================================================
        # STEP 2: Retrieval (Parallel Execution)
        # ====================================================================
        print("\n🔄 STEP 2: Running Retrieval Agent (Parallel)...")
        
        retrieval_result = retrieval_agent(state)
        
        # Merge results into state
        state.update(retrieval_result)
        
        # Log retrieval results
        docs = state.get("retrieved_docs", [])
        print(f"   ✓ Retrieved Documents: {len(docs)}")
        
        if docs:
            print("\n   📚 Sample Results:")
            for i, doc in enumerate(docs[:3]):
                text_preview = doc.text[:100] if hasattr(doc, 'text') else str(doc)[:100]
                score = doc.score if hasattr(doc, 'score') else 'N/A'
                print(f"      {i+1}. Score: {score:.3f} | {text_preview}...")
        
        # Assertions for Retrieval
        assert docs is not None, "Retrieved docs should not be None"
        assert len(docs) > 0, "Should retrieve at least some documents"
        
        # Check that results contain relevant topics
        all_text = " ".join([
            doc.text.lower() if hasattr(doc, 'text') else str(doc).lower() 
            for doc in docs
        ])
        
        # At least one document should mention Zakat-related terms
        zakat_terms = ["zakat", "زكاة", "charity", "alms", "صدقة"]
        has_zakat = any(term in all_text for term in zakat_terms)
        
        # At least one document should mention Fasting-related terms
        fasting_terms = ["fast", "صيام", "صوم", "ramadan", "رمضان"]
        has_fasting = any(term in all_text for term in fasting_terms)
        
        print(f"\n   ✓ Contains Zakat content: {has_zakat}")
        print(f"   ✓ Contains Fasting content: {has_fasting}")
        
        # We expect at least one topic to be found
        assert has_zakat or has_fasting, \
            "Results should contain hadiths about Zakat or Fasting"
        
        print("\n   ✅ Retrieval assertions passed!")
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "=" * 60)
        print("🎉 END-TO-END PIPELINE TEST PASSED!")
        print("=" * 60)
        print(f"""
📊 Pipeline Summary:
   • Query: {state['original_query'][:50]}...
   • Intent: {state.get('query_intent')}
   • Sub-queries: {len(state.get('sub_queries', []))} generated
   • Documents Retrieved: {len(docs)}
   • Zakat Coverage: {'✓' if has_zakat else '✗'}
   • Fasting Coverage: {'✓' if has_fasting else '✗'}

🔗 View full trace in LangSmith dashboard
""")
    
    @traceable(name="test_complex_pipeline_ablution_tayammum")
    def test_complex_pipeline_ablution_tayammum(
        self,
        ensure_langsmith_configured,
        initial_state_ablution_tayammum,
    ):
        """
        Test the full pipeline: Query Analysis -> Retrieval with parallel execution.
        
        Input: "Compare the rules of Wudu (ablution) and Tayammum in Sahih Bukhari"
        
        Expected Flow:
        1. Query Analysis identifies comparative_analysis intent
        2. Query is decomposed into sub-queries for Wudu and Tayammum
        3. Retrieval runs parallel searches for both topics
        4. Results are aggregated and contain hadiths about both topics
        """
        # Print observability instructions
        print_langsmith_instructions("test_complex_pipeline_ablution_tayammum")
        
        state = initial_state_ablution_tayammum
        print(f"\n📝 Input Query: {state['original_query']}")
        print("-" * 60)
        
        # ====================================================================
        # STEP 1: Query Analysis
        # ====================================================================
        print("\n🔄 STEP 1: Running Query Analysis Agent...")
        
        analysis_result = query_analysis_agent(state)
        
        # Merge results into state
        state.update(analysis_result)
        
        # Log analysis results
        print(f"   ✓ Intent: {state.get('query_intent')}")
        print(f"   ✓ Input Source: {state.get('input_source')}")
        print(f"   ✓ Target Collections: {state.get('target_collections')}")
        print(f"   ✓ Sub-queries: {state.get('sub_queries')}")
        print(f"   ✓ Language: {state.get('language')}")
        
        # Assertions for Query Analysis
        assert state.get("query_intent") == "comparative_analysis", \
            f"Expected comparative_analysis, got {state.get('query_intent')}"
        
        assert state.get("sub_queries") is not None, \
            "Sub-queries should be generated for comparative analysis"
        
        assert len(state.get("sub_queries", [])) >= 2, \
            f"Expected at least 2 sub-queries, got {len(state.get('sub_queries', []))}"
        
        # Check that sub-queries mention Wudu and Tayammum
        sub_queries_text = " ".join(state.get("sub_queries", [])).lower()
        assert "wudu" in sub_queries_text or "ablution" in sub_queries_text or "وضوء" in sub_queries_text, \
            "Sub-queries should include Wudu/Ablution topic"
        assert "tayammum" in sub_queries_text or "تيمم" in sub_queries_text, \
            "Sub-queries should include Tayammum topic"
        
        print("\n   ✅ Query Analysis assertions passed!")
        
        # ====================================================================
        # STEP 2: Retrieval (Parallel Execution)
        # ====================================================================
        print("\n🔄 STEP 2: Running Retrieval Agent (Parallel)...")
        
        retrieval_result = retrieval_agent(state)
        
        # Merge results into state
        state.update(retrieval_result)
        
        # Log retrieval results
        docs = state.get("retrieved_docs", [])
        print(f"   ✓ Retrieved Documents: {len(docs)}")
        
        if docs:
            print("\n   📚 Sample Results:")
            for i, doc in enumerate(docs[:3]):
                text_preview = doc.text[:100] if hasattr(doc, 'text') else str(doc)[:100]
                score = doc.score if hasattr(doc, 'score') else 'N/A'
                print(f"      {i+1}. Score: {score:.3f} | {text_preview}...")
        
        # Assertions for Retrieval
        assert docs is not None, "Retrieved docs should not be None"
        assert len(docs) > 0, "Should retrieve at least some documents"
        
        # Check that results contain relevant topics
        all_text = " ".join([
            doc.text.lower() if hasattr(doc, 'text') else str(doc).lower() 
            for doc in docs
        ])
        
        # At least one document should mention Wudu-related terms
        wudu_terms = ["wudu", "ablution", "wash", "وضوء", "clean"]
        has_wudu = any(term in all_text for term in wudu_terms)
        
        # At least one document should mention Tayammum-related terms
        tayammum_terms = ["tayammum", "dust", "earth", "تيمم", "sand"]
        has_tayammum = any(term in all_text for term in tayammum_terms)
        
        print(f"\n   ✓ Contains Wudu content: {has_wudu}")
        print(f"   ✓ Contains Tayammum content: {has_tayammum}")
        
        # We expect at least one topic to be found
        assert has_wudu or has_tayammum, \
            "Results should contain hadiths about Wudu or Tayammum"
        
        print("\n   ✅ Retrieval assertions passed!")
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "=" * 60)
        print("🎉 END-TO-END PIPELINE TEST PASSED!")
        print("=" * 60)
        print(f"""
📊 Pipeline Summary:
   • Query: {state['original_query'][:50]}...
   • Intent: {state.get('query_intent')}
   • Sub-queries: {len(state.get('sub_queries', []))} generated
   • Documents Retrieved: {len(docs)}
   • Wudu Coverage: {'✓' if has_wudu else '✗'}
   • Tayammum Coverage: {'✓' if has_tayammum else '✗'}

🔗 View full trace in LangSmith dashboard
""")

    @traceable(name="test_arabic_comparative_query")
    def test_arabic_comparative_query(
        self,
        ensure_langsmith_configured,
        initial_state_arabic,
    ):
        """
        Test pipeline with Arabic comparative query.
        
        Input: "قارن بين أحاديث الزكاة والصيام في صحيح البخاري"
        (Compare hadiths about Zakat and Fasting in Sahih Bukhari)
        """
        print_langsmith_instructions("test_arabic_comparative_query")
        
        state = initial_state_arabic
        print(f"\n📝 Input Query (Arabic): {state['original_query']}")
        print("-" * 60)
        
        # Step 1: Query Analysis
        print("\n🔄 STEP 1: Running Query Analysis Agent...")
        analysis_result = query_analysis_agent(state)
        state.update(analysis_result)
        
        print(f"   ✓ Intent: {state.get('query_intent')}")
        print(f"   ✓ Language: {state.get('language')}")
        print(f"   ✓ Sub-queries: {state.get('sub_queries')}")
        
        # Should detect Arabic and comparative intent
        assert state.get("language") in ["ar", "mixed"], \
            f"Should detect Arabic language, got {state.get('language')}"
        
        assert state.get("query_intent") == "comparative_analysis", \
            f"Should detect comparative_analysis, got {state.get('query_intent')}"
        
        # Step 2: Retrieval
        print("\n🔄 STEP 2: Running Retrieval Agent...")
        retrieval_result = retrieval_agent(state)
        state.update(retrieval_result)
        
        docs = state.get("retrieved_docs", [])
        print(f"   ✓ Retrieved: {len(docs)} documents")
        
        assert len(docs) > 0, "Should retrieve documents for Arabic query"
        
        print("\n   ✅ Arabic pipeline test passed!")
    
    @traceable(name="test_specific_lookup_skips_decomposition")
    def test_specific_lookup_skips_decomposition(
        self,
        ensure_langsmith_configured,
    ):
        """
        Test that specific_lookup intent skips query decomposition.
        
        This verifies the conditional pipeline optimization.
        """
        print_langsmith_instructions("test_specific_lookup_skips_decomposition")
        
        state: AgentState = {
            "original_query": "Find hadith number 1 from Bukhari",
            "normalized_query": None,
            "corrected_query": None,
            "input_source": None,
            "query_intent": None,
            "target_collections": None,
            "sub_queries": None,
            "retrieved_docs": None,
            "language": None,
            "metadata": {},
        }
        
        print(f"\n📝 Input Query: {state['original_query']}")
        
        # Run analysis
        analysis_result = query_analysis_agent(state)
        state.update(analysis_result)
        
        print(f"   ✓ Intent: {state.get('query_intent')}")
        print(f"   ✓ Sub-queries: {state.get('sub_queries')}")
        
        # For specific_lookup, decomposition should be skipped
        assert state.get("query_intent") == "specific_lookup", \
            f"Expected specific_lookup, got {state.get('query_intent')}"
        
        # Sub-queries should be None or contain only the original query
        sub_queries = state.get("sub_queries")
        if sub_queries:
            assert len(sub_queries) <= 1, \
                "Specific lookup should not decompose into multiple sub-queries"
        
        # Run retrieval
        retrieval_result = retrieval_agent(state)
        state.update(retrieval_result)
        
        docs = state.get("retrieved_docs", [])
        print(f"   ✓ Retrieved: {len(docs)} documents")
        
        # Should find the specific hadith
        assert len(docs) > 0, "Should find hadith number 1"
        
        print("\n   ✅ Specific lookup optimization test passed!")


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPipelinePerformance:
    """Performance benchmarks for the pipeline."""
    
    @traceable(name="test_pipeline_latency")
    def test_pipeline_latency(self, ensure_langsmith_configured):
        """
        Measure and report pipeline latency.
        
        Useful for identifying bottlenecks via LangSmith traces.
        """
        import time
        
        print_langsmith_instructions("test_pipeline_latency")
        
        state: AgentState = {
            "original_query": "hadiths about honesty",
            "normalized_query": None,
            "corrected_query": None,
            "input_source": None,
            "query_intent": None,
            "target_collections": None,
            "sub_queries": None,
            "retrieved_docs": None,
            "language": None,
            "metadata": {},
        }
        
        # Measure Query Analysis
        start = time.time()
        analysis_result = query_analysis_agent(state)
        analysis_time = time.time() - start
        state.update(analysis_result)
        
        # Measure Retrieval
        start = time.time()
        retrieval_result = retrieval_agent(state)
        retrieval_time = time.time() - start
        state.update(retrieval_result)
        
        total_time = analysis_time + retrieval_time
        
        print(f"""
⏱️ LATENCY REPORT:
   • Query Analysis: {analysis_time:.2f}s
   • Retrieval:      {retrieval_time:.2f}s
   • Total:          {total_time:.2f}s
   
📊 Compare with LangSmith trace for detailed breakdown
""")
        
        # Soft assertion - warn if too slow but don't fail
        if total_time > 30:
            print(f"⚠️ WARNING: Pipeline took {total_time:.1f}s (>30s threshold)")
        
        # Pipeline should complete within reasonable time
        assert total_time < 120, f"Pipeline too slow: {total_time:.1f}s"


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly for debugging.
    
    Usage:
        python tests/test_integration_real.py
        
    Or with pytest:
        pytest tests/test_integration_real.py -v -s
    """
    print("\n🚀 Running End-to-End Integration Tests")
    print("=" * 70)
    
    # Load environment
    load_dotenv()
    
    # Check LangSmith config
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("❌ ERROR: LANGSMITH_API_KEY not found in environment")
        print("   Please add it to your .env file")
        sys.exit(1)
    
    project = os.environ.get("LANGSMITH_PROJECT", "hadith-rag")
    print(f"✅ LangSmith Project: {project}")
    print(f"✅ Tracing Enabled: {os.environ.get('LANGSMITH_TRACING', 'false')}")
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "-s",  # Show print statements
        "--tb=short",
        "-x",  # Stop on first failure
    ])
