"""
Test for Arabic Hadith Search - Observable via LangSmith

This test searches for the famous hadith about the beginning of revelation
to observe the full pipeline through LangSmith.

Based on test_complex_pipeline_flow from test_integration_real.py
"""

import os
import sys
import logging
import pytest
from typing import Dict, Any, List
from dotenv import load_dotenv
from langsmith import traceable, Client

# Load environment variables FIRST
load_dotenv()

# Enable LangSmith tracing (use correct env var names)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "hadith-rag")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import state type and agents
from src.graph.state import AgentState
from src.agents.query_analysis import query_analysis_agent
from src.agents.retrieval import retrieval_agent


# ============================================================================
# LangSmith Helper Functions
# ============================================================================

def print_langsmith_instructions(test_name: str):
    """Print step-by-step instructions for viewing traces in LangSmith."""
    project_name = os.environ.get("LANGSMITH_PROJECT", "hadith-rag")
    
    print("\n" + "=" * 70)
    print("🔎 LANGSMITH OBSERVABILITY INSTRUCTIONS")
    print("=" * 70)
    print(f"""
📊 To view this test run in LangSmith:

1. Go to: https://smith.langchain.com/

2. Select your project: '{project_name}'

3. Look for the trace named: '{test_name}'
   - Sort by 'Start Time' (descending) to find the latest run

4. Explore the trace hierarchy:
   ┌─ {test_name} (root)
   │  ├─ query_analysis_agent
   │  │  ├─ input_source_identification_tool
   │  │  ├─ query_normalization_tool
   │  │  ├─ intent_classification_tool
   │  │  ├─ collection_target_detection_tool
   │  │  └─ query_decomposition_tool
   │  │
   │  └─ retrieval_agent
   │     ├─ autonomous_search_loop
   │     │  ├─ agent_decision (LLM decides tool)
   │     │  ├─ agent_tool_* (executes chosen tool)
   │     │  ├─ agent_decision (next action)
   │     │  └─ ... (ReAct loop continues)
   │     └─ aggregate_results

5. Look for PARALLEL execution and ReAct decisions!
""")
    print("=" * 70 + "\n")


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def ensure_langsmith_configured():
    """Verify LangSmith is properly configured before running tests."""
    api_key = os.environ.get("LANGSMITH_API_KEY")
    
    if not api_key:
        pytest.skip("LANGSMITH_API_KEY not set in environment")
    
    assert os.environ.get("LANGSMITH_TRACING") == "true", \
        "LANGSMITH_TRACING should be 'true'"
    
    project = os.environ.get("LANGSMITH_PROJECT", "hadith-rag")
    print(f"\n✅ LangSmith configured: Project='{project}'")
    
    return True


@pytest.fixture
def initial_state_revelation_arabic() -> AgentState:
    """
    Create initial state for Arabic revelation hadith search.
    
    This is a SEARCH QUERY (not hadith text) asking about the revelation.
    """
    return {
        "original_query": "ابحث عن حديث بدء الوحي وكيف نزل جبريل على النبي في غار حراء",
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
def initial_state_revelation_english() -> AgentState:
    """
    Create initial state for English revelation hadith search.
    """
    return {
        "original_query": "Find hadiths about the beginning of revelation and how Gabriel came to Prophet Muhammad in cave Hira",
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
def initial_state_comparative_revelation() -> AgentState:
    """
    Create initial state for comparative analysis about revelation.
    """
    return {
        "original_query": "قارن بين أحاديث بدء الوحي وأحاديث الإسراء والمعراج في صحيح البخاري",
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

class TestArabicHadithPipeline:
    """
    End-to-end integration tests for Arabic hadith search.
    
    These tests use real agents with real LLM calls and vector store.
    Modeled after test_complex_pipeline_flow for proper ReAct observability.
    """
    
    @traceable(name="test_revelation_hadith_arabic")
    def test_revelation_hadith_arabic(
        self,
        ensure_langsmith_configured,
        initial_state_revelation_arabic,
    ):
        """
        Test searching for revelation hadith in Arabic.
        
        Input: "ابحث عن حديث بدء الوحى وكيف نزل جبريل على النبي فى غار حراء"
        (Search for the hadith about the beginning of revelation and how 
        Gabriel came to the Prophet in cave Hira)
        
        Expected Flow:
        1. Query Analysis identifies exploratory_search intent
        2. Retrieval uses autonomous ReAct agent
        3. Agent decides which tools to use (semantic, keyword, hybrid)
        4. Results contain hadiths about revelation
        """
        print_langsmith_instructions("test_revelation_hadith_arabic")
        
        state = initial_state_revelation_arabic
        print(f"\n📝 Input Query: {state['original_query']}")
        print("-" * 60)
        
        # ====================================================================
        # STEP 1: Query Analysis
        # ====================================================================
        print("\n🔄 STEP 1: Running Query Analysis Agent...")
        
        analysis_result = query_analysis_agent(state)
        state.update(analysis_result)
        
        print(f"   ✓ Intent: {state.get('query_intent')}")
        print(f"   ✓ Input Source: {state.get('input_source')}")
        print(f"   ✓ Target Collections: {state.get('target_collections')}")
        print(f"   ✓ Sub-queries: {state.get('sub_queries')}")
        print(f"   ✓ Language: {state.get('language')}")
        
        # Should detect Arabic and base_knowledge (search query, not hadith text)
        assert state.get("language") in ["ar", "mixed"], \
            f"Should detect Arabic language, got {state.get('language')}"
        
        assert state.get("input_source") == "base_knowledge", \
            f"Should detect base_knowledge (search query), got {state.get('input_source')}"
        
        print("\n   ✅ Query Analysis assertions passed!")
        
        # ====================================================================
        # STEP 2: Retrieval (Autonomous ReAct Agent)
        # ====================================================================
        print("\n🔄 STEP 2: Running Retrieval Agent (Autonomous ReAct)...")
        
        retrieval_result = retrieval_agent(state)
        state.update(retrieval_result)
        
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
        
        # Check for revelation-related content
        all_text = " ".join([
            doc.text.lower() if hasattr(doc, 'text') else str(doc).lower() 
            for doc in docs
        ])
        
        revelation_terms_ar = ["الوحي", "حراء", "جبريل", "خديجة", "الرؤيا"]
        revelation_terms_en = ["revelation", "hira", "gabriel", "khadija", "dream"]
        
        has_revelation = (
            any(term in all_text for term in revelation_terms_ar) or
            any(term in all_text for term in revelation_terms_en)
        )
        
        print(f"\n   ✓ Contains revelation content: {has_revelation}")
        
        assert has_revelation, \
            "Results should contain hadiths about revelation"
        
        print("\n   ✅ Retrieval assertions passed!")
        
        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "=" * 60)
        print("🎉 ARABIC REVELATION HADITH TEST PASSED!")
        print("=" * 60)
        print(f"""
📊 Pipeline Summary:
   • Query: {state['original_query'][:50]}...
   • Intent: {state.get('query_intent')}
   • Input Source: {state.get('input_source')}
   • Sub-queries: {len(state.get('sub_queries', []) or [])} generated
   • Documents Retrieved: {len(docs)}

🔗 View full trace in LangSmith dashboard
""")
        
        # Flush logs
        Client().flush()
    
    @traceable(name="test_revelation_hadith_english")
    def test_revelation_hadith_english(
        self,
        ensure_langsmith_configured,
        initial_state_revelation_english,
    ):
        """
        Test searching for revelation hadith in English.
        
        Input: "Find hadiths about the beginning of revelation..."
        """
        print_langsmith_instructions("test_revelation_hadith_english")
        
        state = initial_state_revelation_english
        print(f"\n📝 Input Query: {state['original_query']}")
        print("-" * 60)
        
        # Step 1: Query Analysis
        print("\n🔄 STEP 1: Running Query Analysis Agent...")
        analysis_result = query_analysis_agent(state)
        state.update(analysis_result)
        
        print(f"   ✓ Intent: {state.get('query_intent')}")
        print(f"   ✓ Input Source: {state.get('input_source')}")
        print(f"   ✓ Language: {state.get('language')}")
        
        assert state.get("input_source") == "base_knowledge", \
            f"Should detect base_knowledge, got {state.get('input_source')}"
        
        # Step 2: Retrieval
        print("\n🔄 STEP 2: Running Retrieval Agent...")
        retrieval_result = retrieval_agent(state)
        state.update(retrieval_result)
        
        docs = state.get("retrieved_docs", [])
        print(f"   ✓ Retrieved: {len(docs)} documents")
        
        assert len(docs) > 0, "Should retrieve documents"
        
        # Check for revelation content
        all_text = " ".join([doc.text.lower() for doc in docs])
        has_revelation = any(term in all_text for term in 
                           ["revelation", "hira", "gabriel", "الوحي", "حراء"])
        
        assert has_revelation, "Should find revelation hadiths"
        
        print("\n   ✅ English revelation test passed!")
        Client().flush()
    
    @traceable(name="test_comparative_revelation_isra")
    def test_comparative_revelation_isra(
        self,
        ensure_langsmith_configured,
        initial_state_comparative_revelation,
    ):
        """
        Test comparative analysis: Revelation vs Isra/Miraj hadiths.
        
        Input: "قارن بين أحاديث بدء الوحي وأحاديث الإسراء والمعراج"
        (Compare hadiths about revelation and Isra/Miraj)
        
        This should trigger:
        - comparative_analysis intent
        - Query decomposition into sub-queries
        - Parallel searches
        """
        print_langsmith_instructions("test_comparative_revelation_isra")
        
        state = initial_state_comparative_revelation
        print(f"\n📝 Input Query: {state['original_query']}")
        print("-" * 60)
        
        # Step 1: Query Analysis
        print("\n🔄 STEP 1: Running Query Analysis Agent...")
        analysis_result = query_analysis_agent(state)
        state.update(analysis_result)
        
        print(f"   ✓ Intent: {state.get('query_intent')}")
        print(f"   ✓ Input Source: {state.get('input_source')}")
        print(f"   ✓ Sub-queries: {state.get('sub_queries')}")
        print(f"   ✓ Language: {state.get('language')}")
        
        # Should detect comparative analysis
        assert state.get("query_intent") == "comparative_analysis", \
            f"Expected comparative_analysis, got {state.get('query_intent')}"
        
        assert state.get("sub_queries") is not None, \
            "Should generate sub-queries for comparative analysis"
        
        assert len(state.get("sub_queries", [])) >= 2, \
            f"Expected at least 2 sub-queries, got {len(state.get('sub_queries', []))}"
        
        print("\n   ✅ Query Analysis assertions passed!")
        
        # Step 2: Retrieval (Parallel)
        print("\n🔄 STEP 2: Running Retrieval Agent (Parallel)...")
        retrieval_result = retrieval_agent(state)
        state.update(retrieval_result)
        
        docs = state.get("retrieved_docs", [])
        print(f"   ✓ Retrieved: {len(docs)} documents")
        
        if docs:
            print("\n   📚 Sample Results:")
            for i, doc in enumerate(docs[:3]):
                text_preview = doc.text[:80] if hasattr(doc, 'text') else str(doc)[:80]
                print(f"      {i+1}. {text_preview}...")
        
        assert len(docs) > 0, "Should retrieve documents"
        
        # Check for both topics
        all_text = " ".join([doc.text for doc in docs])
        
        revelation_terms = ["الوحي", "حراء", "revelation", "hira"]
        isra_terms = ["الإسراء", "المعراج", "isra", "miraj", "السماء"]
        
        has_revelation = any(term in all_text for term in revelation_terms)
        has_isra = any(term in all_text for term in isra_terms)
        
        print(f"\n   ✓ Contains Revelation content: {has_revelation}")
        print(f"   ✓ Contains Isra/Miraj content: {has_isra}")
        
        # At least one topic should be found
        assert has_revelation or has_isra, \
            "Should find hadiths about revelation or Isra/Miraj"
        
        print("\n   ✅ Comparative test passed!")
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 COMPARATIVE REVELATION/ISRA TEST PASSED!")
        print("=" * 60)
        print(f"""
📊 Pipeline Summary:
   • Query: {state['original_query'][:40]}...
   • Intent: {state.get('query_intent')}
   • Sub-queries: {len(state.get('sub_queries', []))} generated
   • Documents Retrieved: {len(docs)}
   • Revelation Coverage: {'✓' if has_revelation else '✗'}
   • Isra/Miraj Coverage: {'✓' if has_isra else '✗'}

🔗 View full trace in LangSmith dashboard
""")
        
        Client().flush()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly for debugging.
    
    Usage:
        python tests/test_arabic_hadith_search.py
        
    Or with pytest:
        pytest tests/test_arabic_hadith_search.py -v -s
    """
    print("\n🚀 Running Arabic Hadith Search Tests")
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
        "-s",
        "--tb=short",
        "-x",
    ])
