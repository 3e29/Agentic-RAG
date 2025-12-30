"""
Hard Tests for GTE Embeddings - Arabic and English Queries

This script tests the GTE embeddings on the first 5 batches (250 Bukhari + 100 Muslim = 350 docs).

Tests:
1. Basic Arabic query (should work well)
2. Basic English query (should work well)
3. Cross-lingual: Arabic query for English concept
4. Cross-lingual: English query for Arabic concept
5. Thematic search: Find hadiths about specific topics
6. Hard test: Comparative/complex queries

Usage:
    python tests/test_gte_embeddings.py
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# GTE ChromaDB path (separate from E5)
CHROMA_DB_PATH = project_root / "data" / "chroma_db_gte"

# GTE Embedding endpoint
GTE_ENDPOINT = "https://sazaitet110--gte-multilingual-embeddings-embed.modal.run"


def get_gte_embedding(text: str) -> List[float]:
    """Get GTE embedding for a text."""
    import httpx
    
    response = httpx.post(
        GTE_ENDPOINT,
        json={"texts": [text]},
        timeout=60.0
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def search_gte_collection(
    collection: Any,
    query: str,
    n_results: int = 5
) -> Dict[str, Any]:
    """Search GTE collection with a query."""
    # Get query embedding
    query_embedding = get_gte_embedding(query)
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return {
        "query": query,
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def print_results(results: Dict[str, Any], test_name: str):
    """Print search results nicely."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Query: {results['query']}")
    print(f"Results: {len(results['documents'])} documents")
    print("-"*70)
    
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'], 
        results['metadatas'], 
        results['distances']
    )):
        # Cosine similarity = 1 - distance (for cosine distance)
        similarity = 1 - dist
        print(f"\n[{i+1}] Similarity: {similarity:.4f}")
        print(f"    Language: {meta.get('language', 'unknown')}")
        print(f"    Chapter: {meta.get('chapter_title_en', '')} | {meta.get('chapter_title_ar', '')}")
        print(f"    Hadith ID: {meta.get('hadith_id', 'N/A')}")
        
        # Truncate long texts
        text_preview = doc[:200] + "..." if len(doc) > 200 else doc
        print(f"    Text: {text_preview}")


def run_test(collection, query: str, test_name: str, n_results: int = 5):
    """Run a single test and print results."""
    try:
        results = search_gte_collection(collection, query, n_results)
        print_results(results, test_name)
        return results
    except Exception as e:
        print(f"\n❌ TEST FAILED: {test_name}")
        print(f"   Error: {e}")
        return None


def main():
    print("\n" + "="*70)
    print("GTE EMBEDDING TESTS - Hard Tests in Arabic and English")
    print("Model: Alibaba-NLP/gte-multilingual-base (768 dim)")
    print("="*70)
    
    # Initialize ChromaDB
    if not CHROMA_DB_PATH.exists():
        print(f"❌ ChromaDB not found at {CHROMA_DB_PATH}")
        print("   Run embed_chunks_gte.py first")
        return
    
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # Get collections
    try:
        bukhari_collection = client.get_collection("hadith_bukhari_gte")
        print(f"\n✓ Bukhari collection: {bukhari_collection.count()} documents")
    except Exception as e:
        print(f"❌ Could not get Bukhari collection: {e}")
        return
    
    try:
        muslim_collection = client.get_collection("hadith_muslim_gte")
        print(f"✓ Muslim collection: {muslim_collection.count()} documents")
    except Exception as e:
        print(f"⚠ Muslim collection not available: {e}")
        muslim_collection = None
    
    # Use Bukhari for main tests (250 docs embedded)
    collection = bukhari_collection
    
    # =========================================================================
    # TEST 1: Basic Arabic Query - Faith/Iman
    # =========================================================================
    run_test(
        collection,
        query="الإيمان بالله",
        test_name="1. Basic Arabic Query - Faith in Allah",
        n_results=5
    )
    
    # =========================================================================
    # TEST 2: Basic English Query - Faith
    # =========================================================================
    run_test(
        collection,
        query="faith and belief in Islam",
        test_name="2. Basic English Query - Faith and Belief",
        n_results=5
    )
    
    # =========================================================================
    # TEST 3: Cross-lingual - Arabic query for English concept
    # =========================================================================
    run_test(
        collection,
        query="أركان الإسلام الخمسة",
        test_name="3. Arabic Query - Five Pillars of Islam",
        n_results=5
    )
    
    # =========================================================================
    # TEST 4: Cross-lingual - English query for Arabic concept
    # =========================================================================
    run_test(
        collection,
        query="The five pillars of Islam shahada prayer zakat fasting hajj",
        test_name="4. English Query - Five Pillars of Islam",
        n_results=5
    )
    
    # =========================================================================
    # TEST 5: Thematic - Prayer/Salah
    # =========================================================================
    run_test(
        collection,
        query="الصلاة وفضلها",
        test_name="5. Arabic Query - Prayer and its Virtues",
        n_results=5
    )
    
    # =========================================================================
    # TEST 6: Thematic - Prayer (English)
    # =========================================================================
    run_test(
        collection,
        query="prayer times and importance of salah",
        test_name="6. English Query - Prayer Times and Importance",
        n_results=5
    )
    
    # =========================================================================
    # TEST 7: HARD - Angel Gabriel hadith
    # =========================================================================
    run_test(
        collection,
        query="جبريل يسأل النبي عن الإسلام والإيمان والإحسان",
        test_name="7. HARD Arabic - Gabriel Asking About Islam/Iman/Ihsan",
        n_results=5
    )
    
    # =========================================================================
    # TEST 8: HARD - Gabriel hadith (English)
    # =========================================================================
    run_test(
        collection,
        query="Angel Gabriel came to the Prophet and asked about Islam, Iman, and Ihsan",
        test_name="8. HARD English - Gabriel Hadith",
        n_results=5
    )
    
    # =========================================================================
    # TEST 9: HARD - Best deeds
    # =========================================================================
    run_test(
        collection,
        query="ما أفضل الأعمال عند الله",
        test_name="9. HARD Arabic - Best Deeds",
        n_results=5
    )
    
    # =========================================================================
    # TEST 10: HARD - Best deeds (English)
    # =========================================================================
    run_test(
        collection,
        query="What are the best deeds most beloved to Allah",
        test_name="10. HARD English - Best Deeds",
        n_results=5
    )
    
    # =========================================================================
    # TEST 11: HARD - Signs of hypocrisy
    # =========================================================================
    run_test(
        collection,
        query="علامات المنافق الكذب وخلف الوعد",
        test_name="11. HARD Arabic - Signs of Hypocrisy",
        n_results=5
    )
    
    # =========================================================================
    # TEST 12: HARD - Hypocrisy (English)
    # =========================================================================
    run_test(
        collection,
        query="Signs of a hypocrite: lying, breaking promises, betraying trust",
        test_name="12. HARD English - Signs of Hypocrisy",
        n_results=5
    )
    
    # =========================================================================
    # TEST 13: EXTREME - Comparative concept
    # =========================================================================
    run_test(
        collection,
        query="الفرق بين الإسلام والإيمان",
        test_name="13. EXTREME Arabic - Difference between Islam and Iman",
        n_results=5
    )
    
    # =========================================================================
    # TEST 14: EXTREME - Shame/Modesty (Haya)
    # =========================================================================
    run_test(
        collection,
        query="الحياء شعبة من الإيمان",
        test_name="14. HARD Arabic - Modesty is a Branch of Faith",
        n_results=5
    )
    
    # =========================================================================
    # TEST 15: EXTREME - Modesty (English)
    # =========================================================================
    run_test(
        collection,
        query="modesty shyness haya is part of faith iman",
        test_name="15. HARD English - Modesty and Faith",
        n_results=5
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total documents in Bukhari GTE: {bukhari_collection.count()}")
    if muslim_collection:
        print(f"Total documents in Muslim GTE: {muslim_collection.count()}")
    print(f"Embedding model: Alibaba-NLP/gte-multilingual-base")
    print(f"Embedding dimension: 768")
    print(f"ChromaDB path: {CHROMA_DB_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()
