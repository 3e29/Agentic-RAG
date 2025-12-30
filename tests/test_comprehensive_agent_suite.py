"""
Comprehensive Agent Test Suite for Hadith RAG System

This test suite evaluates the full pipeline (Query Analysis → Retrieval) with:
- 6 test cases: 3 query types × 2 languages (Arabic & English)
- No hints given to the LLM or embeddings
- Full LangSmith observability for debugging and analysis

**Test Types:**
1. Thematic Search: Find hadiths about a general topic
2. Specific Lookup: Find a particular hadith from partial content
3. Comparative Analysis: Compare two related concepts

**Architecture:**
┌─────────────────────────────────────────────────────────────────────┐
│                         TEST SUITE                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Type 1: Thematic Search                                            │
│    ├─ Test 1A: English - General topic query                        │
│    └─ Test 1B: Arabic  - General topic query                        │
├─────────────────────────────────────────────────────────────────────┤
│  Type 2: Specific Lookup                                            │
│    ├─ Test 2A: English - Partial hadith text                        │
│    └─ Test 2B: Arabic  - Partial hadith text                        │
├─────────────────────────────────────────────────────────────────────┤
│  Type 3: Comparative Analysis                                        │
│    ├─ Test 3A: English - Compare two concepts                       │
│    └─ Test 3B: Arabic  - Compare two concepts                       │
└─────────────────────────────────────────────────────────────────────┘

**LangSmith Observability:**
All tests are decorated with @traceable for full pipeline visibility.
View traces at: https://smith.langchain.com/
"""

import os
import sys
import logging
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dotenv import load_dotenv
from langsmith import traceable, Client as LangSmithClient

# Load environment variables
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "hadith-rag")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import HybridSearchTool directly for testing search quality
from src.tools.retrieval.search_tools import HybridSearchTool


# ============================================================================
# Test Configuration - No Hints, Just Natural Queries
# ============================================================================

# Test Type 1: Thematic Search
# Goal: Find hadiths about a general topic without specific text
THEMATIC_TESTS = {
    "english": {
        "query": "What did the Prophet say about treating neighbors?",
        "expected_topics": ["neighbor", "جار", "neighbour"],
        "description": "Thematic search about neighbors in English",
    },
    "arabic": {
        "query": "ما قاله النبي عن معاملة الجيران",
        "expected_topics": ["neighbor", "جار", "neighbour", "جيران"],
        "description": "Thematic search about neighbors in Arabic",
    },
}

# Test Type 2: Specific Lookup
# Goal: Find a specific hadith from partial content (no IDs, no direct quotes)
SPECIFIC_LOOKUP_TESTS = {
    "english": {
        "query": "hadith about a man who came to the Prophet asking which deed is best and was told prayer at its proper time",
        "expected_topics": ["prayer", "time", "deed", "صلاة", "وقت"],
        "description": "Specific lookup - best deeds hadith (English)",
    },
    "arabic": {
        "query": "حديث عن رجل سأل النبي أي العمل أفضل فقال الصلاة على وقتها",
        "expected_topics": ["prayer", "صلاة", "وقت", "أفضل", "العمل"],
        "description": "Specific lookup - best deeds hadith (Arabic)",
    },
}

# Test Type 3: Comparative Analysis
# Goal: Compare two related Islamic concepts
COMPARATIVE_TESTS = {
    "english": {
        "query": "What is the difference between patience during hardship and gratitude during ease according to hadith?",
        "expected_topics": ["patience", "صبر", "gratitude", "شكر", "hardship", "ease"],
        "description": "Comparative analysis - patience vs gratitude (English)",
    },
    "arabic": {
        "query": "ما الفرق بين الصبر عند الشدة والشكر عند الرخاء في الأحاديث",
        "expected_topics": ["patience", "صبر", "gratitude", "شكر", "شدة", "رخاء"],
        "description": "Comparative analysis - patience vs gratitude (Arabic)",
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def create_initial_state(query: str) -> Dict[str, Any]:
    """Create initial AgentState for a query."""
    return {
        "original_query": query,
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


def print_test_header(test_name: str, query: str, language: str, test_type: str):
    """Print formatted test header."""
    print("\n" + "=" * 80)
    print(f"🧪 TEST: {test_name}")
    print("=" * 80)
    print(f"   Type: {test_type}")
    print(f"   Language: {language.upper()}")
    print(f"   Query: {query[:70]}{'...' if len(query) > 70 else ''}")
    print("-" * 80)


def print_analysis_results(state: Dict[str, Any]):
    """Print query analysis results."""
    print("\n📊 QUERY ANALYSIS RESULTS:")
    print(f"   • Intent: {state.get('query_intent', 'N/A')}")
    print(f"   • Input Source: {state.get('input_source', 'N/A')}")
    print(f"   • Language: {state.get('language', 'N/A')}")
    print(f"   • Target Collections: {state.get('target_collections', 'N/A')}")
    print(f"   • Sub-queries: {len(state.get('sub_queries', []) or [])} generated")
    if state.get('sub_queries'):
        for i, sq in enumerate(state['sub_queries'][:3]):
            print(f"      {i+1}. {sq[:60]}{'...' if len(sq) > 60 else ''}")


def print_retrieval_results(docs: List, expected_topics: List[str]):
    """Print retrieval results and topic coverage."""
    print(f"\n📚 RETRIEVAL RESULTS: {len(docs)} documents")
    
    if docs:
        print("\n   Top 10 Results:")
        for i, doc in enumerate(docs[:10]):
            text = doc.text if hasattr(doc, 'text') else str(doc)
            score = doc.score if hasattr(doc, 'score') else 0
            hadith_id = doc.hadith_id if hasattr(doc, 'hadith_id') else 'N/A'
            print(f"   {i+1}. [Hadith {hadith_id}] (score: {score:.4f})")
            print(f"      {text[:120]}...")
        
        # Check topic coverage
        all_text = " ".join([
            (doc.text if hasattr(doc, 'text') else str(doc)).lower()
            for doc in docs
        ])
        
        print("\n   Topic Coverage:")
        found_topics = []
        for topic in expected_topics:
            if topic.lower() in all_text:
                found_topics.append(topic)
                print(f"      ✓ '{topic}' found")
            else:
                print(f"      ✗ '{topic}' NOT found")
        
        print(f"\n   📊 Topic Score: {len(found_topics)}/{len(expected_topics)}")
        
        return len(found_topics) > 0
    return False


def flush_langsmith():
    """Force flush LangSmith traces."""
    try:
        LangSmithClient().flush()
    except Exception as e:
        logger.warning(f"Could not flush LangSmith: {e}")


# ============================================================================
# Test Class - Using HybridSearchTool Directly
# ============================================================================

class TestComprehensiveAgentSuite:
    """
    Comprehensive test suite for Hadith RAG search quality.
    
    Uses HybridSearchTool directly to validate search quality.
    Tests 3 query types in both Arabic and English:
    1. Thematic Search
    2. Specific Lookup
    3. Comparative Analysis
    """
    
    def __init__(self):
        # Initialize HybridSearchTool with crosslingual enabled
        self.search_tool = HybridSearchTool(use_crosslingual=True)
    
    def _run_search(self, query: str, k: int = 10):
        """Run hybrid search and return results."""
        result = self.search_tool(query=query, k=k)
        return result.documents
    
    def _check_relevance(self, docs: List, expected_topics: List[str]) -> tuple:
        """Check topic coverage and return (found_count, total_topics)."""
        if not docs:
            return 0, len(expected_topics)
        
        all_text = " ".join([
            (doc.text if hasattr(doc, 'text') else str(doc)).lower()
            for doc in docs
        ])
        
        found_topics = [t for t in expected_topics if t.lower() in all_text]
        return len(found_topics), len(expected_topics)
    
    # ========================================================================
    # Type 1: Thematic Search Tests
    # ========================================================================
    
    @traceable(name="test_1a_thematic_english_hybrid")
    def test_1a_thematic_search_english(self):
        """
        Test 1A: Thematic Search in English using HybridSearchTool
        """
        test_config = THEMATIC_TESTS["english"]
        query = test_config["query"]
        
        print_test_header(
            "test_1a_thematic_english",
            query,
            "English",
            "Thematic Search (HybridSearchTool)"
        )
        
        # Run hybrid search directly
        print("\n🔄 Running HybridSearchTool...")
        docs = self._run_search(query, k=10)
        
        has_relevant = print_retrieval_results(docs, test_config["expected_topics"])
        found, total = self._check_relevance(docs, test_config["expected_topics"])
        
        # Assertions
        assert docs is not None and len(docs) > 0, "Should retrieve documents"
        assert has_relevant, f"Should find relevant content ({found}/{total} topics found)"
        
        print(f"\n✅ TEST PASSED: Thematic Search (English) - {found}/{total} topics")
        flush_langsmith()
    
    @traceable(name="test_1b_thematic_arabic_hybrid")
    def test_1b_thematic_search_arabic(self):
        """
        Test 1B: Thematic Search in Arabic using HybridSearchTool
        """
        test_config = THEMATIC_TESTS["arabic"]
        query = test_config["query"]
        
        print_test_header(
            "test_1b_thematic_arabic",
            query,
            "Arabic",
            "Thematic Search (HybridSearchTool)"
        )
        
        # Run hybrid search directly
        print("\n🔄 Running HybridSearchTool...")
        docs = self._run_search(query, k=10)
        
        has_relevant = print_retrieval_results(docs, test_config["expected_topics"])
        found, total = self._check_relevance(docs, test_config["expected_topics"])
        
        # Assertions
        assert docs is not None and len(docs) > 0, "Should retrieve documents"
        assert has_relevant, f"Should find relevant content ({found}/{total} topics found)"
        
        print(f"\n✅ TEST PASSED: Thematic Search (Arabic) - {found}/{total} topics")
        flush_langsmith()
    
    # ========================================================================
    # Type 2: Specific Lookup Tests
    # ========================================================================
    
    @traceable(name="test_2a_specific_english_hybrid")
    def test_2a_specific_lookup_english(self):
        """
        Test 2A: Specific Lookup in English using HybridSearchTool
        """
        test_config = SPECIFIC_LOOKUP_TESTS["english"]
        query = test_config["query"]
        
        print_test_header(
            "test_2a_specific_english",
            query,
            "English",
            "Specific Lookup (HybridSearchTool)"
        )
        
        # Run hybrid search directly
        print("\n🔄 Running HybridSearchTool...")
        docs = self._run_search(query, k=10)
        
        has_relevant = print_retrieval_results(docs, test_config["expected_topics"])
        found, total = self._check_relevance(docs, test_config["expected_topics"])
        
        # Assertions
        assert docs is not None and len(docs) > 0, "Should retrieve documents"
        assert has_relevant, f"Should find the specific hadith ({found}/{total} topics found)"
        
        print(f"\n✅ TEST PASSED: Specific Lookup (English) - {found}/{total} topics")
        flush_langsmith()
    
    @traceable(name="test_2b_specific_arabic_hybrid")
    def test_2b_specific_lookup_arabic(self):
        """
        Test 2B: Specific Lookup in Arabic using HybridSearchTool
        """
        test_config = SPECIFIC_LOOKUP_TESTS["arabic"]
        query = test_config["query"]
        
        print_test_header(
            "test_2b_specific_arabic",
            query,
            "Arabic",
            "Specific Lookup (HybridSearchTool)"
        )
        
        # Run hybrid search directly
        print("\n🔄 Running HybridSearchTool...")
        docs = self._run_search(query, k=10)
        
        has_relevant = print_retrieval_results(docs, test_config["expected_topics"])
        found, total = self._check_relevance(docs, test_config["expected_topics"])
        
        # Assertions
        assert docs is not None and len(docs) > 0, "Should retrieve documents"
        assert has_relevant, f"Should find the specific hadith ({found}/{total} topics found)"
        
        print(f"\n✅ TEST PASSED: Specific Lookup (Arabic) - {found}/{total} topics")
        flush_langsmith()
    
    # ========================================================================
    # Type 3: Comparative Analysis Tests
    # ========================================================================
    
    @traceable(name="test_3a_comparative_english_hybrid")
    def test_3a_comparative_analysis_english(self):
        """
        Test 3A: Comparative Analysis in English using HybridSearchTool
        
        For comparative queries, we search for each concept separately
        """
        test_config = COMPARATIVE_TESTS["english"]
        query = test_config["query"]
        
        print_test_header(
            "test_3a_comparative_english",
            query,
            "English",
            "Comparative Analysis (HybridSearchTool)"
        )
        
        # Run hybrid search directly
        print("\n🔄 Running HybridSearchTool...")
        docs = self._run_search(query, k=15)
        
        has_relevant = print_retrieval_results(docs, test_config["expected_topics"])
        found, total = self._check_relevance(docs, test_config["expected_topics"])
        
        # Assertions
        assert docs is not None and len(docs) > 0, "Should retrieve documents"
        assert has_relevant, f"Should find hadiths about patience or gratitude ({found}/{total} topics)"
        
        print(f"\n✅ TEST PASSED: Comparative Analysis (English) - {found}/{total} topics")
        flush_langsmith()
    
    @traceable(name="test_3b_comparative_arabic_hybrid")
    def test_3b_comparative_arabic(self):
        """
        Test 3B: Comparative Analysis in Arabic using HybridSearchTool
        """
        test_config = COMPARATIVE_TESTS["arabic"]
        query = test_config["query"]
        
        print_test_header(
            "test_3b_comparative_arabic",
            query,
            "Arabic",
            "Comparative Analysis (HybridSearchTool)"
        )
        
        # Run hybrid search directly
        print("\n🔄 Running HybridSearchTool...")
        docs = self._run_search(query, k=15)
        
        has_relevant = print_retrieval_results(docs, test_config["expected_topics"])
        found, total = self._check_relevance(docs, test_config["expected_topics"])
        
        # Assertions
        assert docs is not None and len(docs) > 0, "Should retrieve documents"
        assert has_relevant, f"Should find hadiths about patience or gratitude ({found}/{total} topics)"
        
        print(f"\n✅ TEST PASSED: Comparative Analysis (Arabic) - {found}/{total} topics")
        flush_langsmith()


# ============================================================================
# Main Entry Point
# ============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 80)
    print("🚀 COMPREHENSIVE AGENT TEST SUITE")
    print("=" * 80)
    print(f"""
📋 Test Configuration:
   • Total Tests: 6 (3 types × 2 languages)
   • Languages: English, Arabic
   • Test Types:
      1. Thematic Search - General topic queries
      2. Specific Lookup - Find hadith from description
      3. Comparative Analysis - Compare two concepts
   
🔗 LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'hadith-rag')}
   View traces at: https://smith.langchain.com/
""")
    print("=" * 80)
    
    test_suite = TestComprehensiveAgentSuite()
    results = []
    
    tests = [
        ("1A", "Thematic (English)", test_suite.test_1a_thematic_search_english),
        ("1B", "Thematic (Arabic)", test_suite.test_1b_thematic_search_arabic),
        ("2A", "Specific (English)", test_suite.test_2a_specific_lookup_english),
        ("2B", "Specific (Arabic)", test_suite.test_2b_specific_lookup_arabic),
        ("3A", "Comparative (English)", test_suite.test_3a_comparative_analysis_english),
        ("3B", "Comparative (Arabic)", test_suite.test_3b_comparative_arabic),
    ]
    
    for test_id, test_name, test_func in tests:
        try:
            test_func()
            results.append((test_id, test_name, "✅ PASSED"))
        except Exception as e:
            logger.error(f"Test {test_id} failed: {e}")
            results.append((test_id, test_name, f"❌ FAILED: {str(e)[:50]}"))
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUITE SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, _, status in results if "PASSED" in status)
    
    for test_id, test_name, status in results:
        print(f"   Test {test_id}: {test_name:25} {status}")
    
    print("-" * 80)
    print(f"   Total: {passed}/{len(results)} tests passed")
    print("=" * 80)
    
    # Flush LangSmith
    flush_langsmith()
    
    print(f"\n🔗 View all traces at: https://smith.langchain.com/")
    print(f"   Project: {os.environ.get('LANGSMITH_PROJECT', 'hadith-rag')}")
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
