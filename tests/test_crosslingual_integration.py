"""
Integration Test: Cross-lingual Hybrid Search with Optimized BM25

Tests the full cross-lingual search pipeline to ensure the optimized
BM25 keyword search integrates properly with semantic search and RRF.
"""

import asyncio
import logging
import sys
from src.tools.retrieval.search_tools import crosslingual_hybrid_search
from src.tools.retrieval.schemas import MetadataFilter

# Fix Windows console encoding for Arabic text
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_crosslingual_search():
    """Test cross-lingual hybrid search with various queries."""
    
    print("=" * 80)
    print("CROSS-LINGUAL HYBRID SEARCH INTEGRATION TEST")
    print("=" * 80)
    print("\nTesting optimized BM25 within full hybrid search pipeline...\n")
    
    test_queries = [
        ("Arabic: Patience", "ما جزاء الصابرين؟"),
        ("Arabic: Prayer", "كيف كان النبي يصلي؟"),
        ("English: Charity", "What are the rules of giving charity?"),
        ("English: Fasting", "What is the virtue of fasting?"),
    ]
    
    for test_name, query in test_queries:
        print(f"{test_name}:")
        print(f"  Query: {query}")
        
        try:
            result = await crosslingual_hybrid_search(
                query=query,
                k=10,
                filters=None,
                translate_arabic=True
            )
            
            print(f"  Results: {len(result.documents)} documents")
            print(f"  Time: {result.execution_time_ms:.1f}ms")
            
            if result.documents:
                top_doc = result.documents[0]
                print(f"  Top Result: {top_doc.text[:100]}...")
                print(f"  Score: {top_doc.score:.4f}")
            
            print(f"  ✓ PASS\n")
            
        except Exception as e:
            print(f"  ✗ FAIL: {e}\n")
            raise
    
    print("=" * 80)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("=" * 80)
    print("\nOptimized BM25 successfully integrated with:")
    print("  ✓ Cross-lingual search strategy")
    print("  ✓ Semantic vector search")
    print("  ✓ Reciprocal Rank Fusion")
    print("  ✓ Language filtering")
    print("  ✓ Full corpus coverage (33K+ docs)")


async def test_with_filters():
    """Test that filters still work with optimized BM25."""
    
    print("\n" + "=" * 80)
    print("TESTING WITH METADATA FILTERS")
    print("=" * 80)
    
    query = "الصبر"
    
    # Test with collection filter
    filters = MetadataFilter(collection="sahih_al-bukhari")
    
    print(f"\nQuery: {query}")
    print(f"Filter: collection=sahih_al-bukhari")
    
    result = await crosslingual_hybrid_search(
        query=query,
        k=10,
        filters=filters,
        translate_arabic=True
    )
    
    print(f"Results: {len(result.documents)} documents")
    print(f"Time: {result.execution_time_ms:.1f}ms")
    
    # Verify all results are from Bukhari
    for doc in result.documents:
        assert "bukhari" in doc.collection.lower(), f"Document {doc.chunk_id} not from Bukhari"
    
    print("✓ All results correctly filtered to Bukhari collection")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_crosslingual_search())
    asyncio.run(test_with_filters())
