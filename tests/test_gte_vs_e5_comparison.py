"""
GTE vs E5 Embedding Comparison Tests

This script compares the GTE and E5 embeddings on the same queries
to see which model performs better for Arabic hadith search.

Test Matrix:
- Arabic queries (basic and hard)
- English queries (basic and hard)
- Cross-lingual queries

Success Criteria:
- For each test, check if the CORRECT hadith appears in top-5 results
- Compare similarity scores between GTE and E5
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
import httpx

logging.basicConfig(level=logging.WARNING)

# Endpoints
GTE_ENDPOINT = "https://sazaitet110--gte-multilingual-embeddings-embed.modal.run"
E5_ENDPOINT = "https://sazaitet110--multilingual-e5-embeddings-embed.modal.run"

# ChromaDB paths
GTE_CHROMA = project_root / "data" / "chroma_db_gte"
E5_CHROMA = project_root / "data" / "chroma_db"


def get_embedding(text: str, endpoint: str) -> List[float]:
    """Get embedding from specified endpoint."""
    response = httpx.post(endpoint, json={"texts": [text]}, timeout=60.0)
    response.raise_for_status()
    return response.json()["embeddings"][0]


def search(collection, query_embedding: List[float], n: int = 5) -> Dict:
    """Search collection with embedding."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    return results


def check_hadith_in_results(results: Dict, target_hadith_id: int) -> Tuple[bool, int, float]:
    """
    Check if target hadith is in results.
    Returns: (found, position, similarity)
    """
    if not results["metadatas"] or not results["metadatas"][0]:
        return False, -1, 0.0
    
    for i, meta in enumerate(results["metadatas"][0]):
        if meta.get("hadith_id") == target_hadith_id:
            similarity = 1 - results["distances"][0][i]
            return True, i + 1, similarity
    
    return False, -1, 0.0


# ============================================================================
# TEST CASES - Each has query, expected hadith ID, and description
# ============================================================================

TEST_CASES = [
    # ========== BASIC TESTS ==========
    {
        "name": "Five Pillars (English)",
        "query_en": "Islam is based on five pillars shahada prayer zakat fasting hajj",
        "query_ar": "أركان الإسلام الخمسة الشهادة الصلاة الزكاة الصيام الحج",
        "expected_hadith_id": 8,  # Famous five pillars hadith
        "description": "Find the hadith about five pillars of Islam"
    },
    {
        "name": "Signs of Hypocrite",
        "query_en": "Signs of a hypocrite: lying, breaking promises, betraying trust",
        "query_ar": "علامات المنافق الكذب وخلف الوعد والخيانة",
        "expected_hadith_id": 33,  # Three signs of hypocrite
        "description": "Find the hadith about three signs of a hypocrite"
    },
    {
        "name": "Faith Branches (Haya)",
        "query_en": "Faith has over sixty branches and haya modesty is part of faith",
        "query_ar": "الإيمان بضع وستون شعبة والحياء شعبة من الإيمان",
        "expected_hadith_id": 9,  # Faith is 60+ branches, haya is part of it
        "description": "Find the hadith about branches of faith"
    },
    {
        "name": "Best Deeds",
        "query_en": "What is the best deed? Belief in Allah, then jihad, then Hajj",
        "query_ar": "أي العمل أفضل إيمان بالله ورسوله ثم الجهاد ثم الحج",
        "expected_hadith_id": 26,  # Best deeds hadith
        "description": "Find the hadith about the best deeds"
    },
    {
        "name": "Gabriel Hadith (Iman)",
        "query_en": "Gabriel asked the Prophet about faith, Islam, and Ihsan excellence",
        "query_ar": "جبريل يسأل النبي عن الإيمان والإسلام والإحسان",
        "expected_hadith_id": 50,  # Gabriel hadith
        "description": "Find the famous Gabriel hadith"
    },
    {
        "name": "Love Prophet More",
        "query_en": "None of you has faith until he loves me more than father and children",
        "query_ar": "لا يؤمن أحدكم حتى أكون أحب إليه من والده وولده",
        "expected_hadith_id": 15,  # Love Prophet more hadith
        "description": "Find the hadith about loving the Prophet"
    },
    {
        "name": "Sweetness of Faith",
        "query_en": "Three qualities bring sweetness of faith: love Allah and Prophet most",
        "query_ar": "ثلاث من كن فيه وجد حلاوة الإيمان",
        "expected_hadith_id": 16,  # Sweetness of faith hadith
        "description": "Find the hadith about sweetness of faith"
    },
    {
        "name": "Wish for Brother",
        "query_en": "None has faith until he wishes for his brother what he wishes for himself",
        "query_ar": "لا يؤمن أحدكم حتى يحب لأخيه ما يحب لنفسه",
        "expected_hadith_id": 13,  # Wish for brother hadith
        "description": "Find the hadith about wishing for others"
    },
]


def run_comparison():
    """Run comparison between GTE and E5."""
    print("\n" + "="*80)
    print("GTE vs E5 EMBEDDING COMPARISON TEST")
    print("="*80)
    
    # Initialize collections
    gte_client = chromadb.PersistentClient(path=str(GTE_CHROMA))
    e5_client = chromadb.PersistentClient(path=str(E5_CHROMA))
    
    try:
        gte_collection = gte_client.get_collection("hadith_bukhari_gte")
        print(f"✓ GTE Collection: {gte_collection.count()} documents")
    except Exception as e:
        print(f"✗ GTE Collection not found: {e}")
        return
    
    try:
        e5_collection = e5_client.get_collection("hadith_bukhari")
        print(f"✓ E5 Collection: {e5_collection.count()} documents")
    except Exception as e:
        print(f"✗ E5 Collection not found: {e}")
        return
    
    # Results tracking
    gte_wins = 0
    e5_wins = 0
    ties = 0
    
    results_table = []
    
    for test in TEST_CASES:
        print(f"\n{'─'*80}")
        print(f"TEST: {test['name']}")
        print(f"Target Hadith ID: {test['expected_hadith_id']}")
        print(f"{'─'*80}")
        
        # Test with English query
        print("\n📝 ENGLISH Query:")
        print(f"   \"{test['query_en'][:60]}...\"")
        
        # GTE English
        try:
            gte_emb_en = get_embedding(test['query_en'], GTE_ENDPOINT)
            gte_results_en = search(gte_collection, gte_emb_en)
            gte_found_en, gte_pos_en, gte_sim_en = check_hadith_in_results(
                gte_results_en, test['expected_hadith_id']
            )
        except Exception as e:
            print(f"   GTE Error: {e}")
            gte_found_en, gte_pos_en, gte_sim_en = False, -1, 0.0
        
        # E5 English
        try:
            e5_emb_en = get_embedding(test['query_en'], E5_ENDPOINT)
            e5_results_en = search(e5_collection, e5_emb_en)
            e5_found_en, e5_pos_en, e5_sim_en = check_hadith_in_results(
                e5_results_en, test['expected_hadith_id']
            )
        except Exception as e:
            print(f"   E5 Error: {e}")
            e5_found_en, e5_pos_en, e5_sim_en = False, -1, 0.0
        
        gte_status_en = f"✓ #{gte_pos_en} (sim={gte_sim_en:.3f})" if gte_found_en else "✗ Not in top 5"
        e5_status_en = f"✓ #{e5_pos_en} (sim={e5_sim_en:.3f})" if e5_found_en else "✗ Not in top 5"
        
        print(f"   GTE: {gte_status_en}")
        print(f"   E5:  {e5_status_en}")
        
        # Test with Arabic query
        print("\n📝 ARABIC Query:")
        print(f"   \"{test['query_ar'][:60]}...\"")
        
        # GTE Arabic
        try:
            gte_emb_ar = get_embedding(test['query_ar'], GTE_ENDPOINT)
            gte_results_ar = search(gte_collection, gte_emb_ar)
            gte_found_ar, gte_pos_ar, gte_sim_ar = check_hadith_in_results(
                gte_results_ar, test['expected_hadith_id']
            )
        except Exception as e:
            print(f"   GTE Error: {e}")
            gte_found_ar, gte_pos_ar, gte_sim_ar = False, -1, 0.0
        
        # E5 Arabic
        try:
            e5_emb_ar = get_embedding(test['query_ar'], E5_ENDPOINT)
            e5_results_ar = search(e5_collection, e5_emb_ar)
            e5_found_ar, e5_pos_ar, e5_sim_ar = check_hadith_in_results(
                e5_results_ar, test['expected_hadith_id']
            )
        except Exception as e:
            print(f"   E5 Error: {e}")
            e5_found_ar, e5_pos_ar, e5_sim_ar = False, -1, 0.0
        
        gte_status_ar = f"✓ #{gte_pos_ar} (sim={gte_sim_ar:.3f})" if gte_found_ar else "✗ Not in top 5"
        e5_status_ar = f"✓ #{e5_pos_ar} (sim={e5_sim_ar:.3f})" if e5_found_ar else "✗ Not in top 5"
        
        print(f"   GTE: {gte_status_ar}")
        print(f"   E5:  {e5_status_ar}")
        
        # Score this test
        gte_score = (1 if gte_found_en else 0) + (1 if gte_found_ar else 0)
        e5_score = (1 if e5_found_en else 0) + (1 if e5_found_ar else 0)
        
        if gte_score > e5_score:
            gte_wins += 1
            winner = "GTE"
        elif e5_score > gte_score:
            e5_wins += 1
            winner = "E5"
        else:
            ties += 1
            winner = "TIE"
        
        print(f"\n   🏆 Winner: {winner} (GTE:{gte_score}/2, E5:{e5_score}/2)")
        
        results_table.append({
            "test": test['name'],
            "gte_en": gte_found_en,
            "gte_ar": gte_found_ar,
            "e5_en": e5_found_en,
            "e5_ar": e5_found_ar,
            "winner": winner
        })
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\n{'Test Name':<30} {'GTE-EN':<10} {'GTE-AR':<10} {'E5-EN':<10} {'E5-AR':<10} {'Winner':<10}")
    print("-"*80)
    
    for r in results_table:
        print(f"{r['test']:<30} "
              f"{'✓' if r['gte_en'] else '✗':<10} "
              f"{'✓' if r['gte_ar'] else '✗':<10} "
              f"{'✓' if r['e5_en'] else '✗':<10} "
              f"{'✓' if r['e5_ar'] else '✗':<10} "
              f"{r['winner']:<10}")
    
    print("-"*80)
    
    total_tests = len(TEST_CASES)
    print(f"\n🏆 OVERALL WINNER:")
    print(f"   GTE Wins: {gte_wins}/{total_tests}")
    print(f"   E5 Wins:  {e5_wins}/{total_tests}")
    print(f"   Ties:     {ties}/{total_tests}")
    
    # Calculate accuracy
    gte_total = sum(1 for r in results_table if r['gte_en']) + sum(1 for r in results_table if r['gte_ar'])
    e5_total = sum(1 for r in results_table if r['e5_en']) + sum(1 for r in results_table if r['e5_ar'])
    max_possible = total_tests * 2  # EN + AR for each test
    
    print(f"\n📊 ACCURACY (finding target hadith in top 5):")
    print(f"   GTE: {gte_total}/{max_possible} = {100*gte_total/max_possible:.1f}%")
    print(f"   E5:  {e5_total}/{max_possible} = {100*e5_total/max_possible:.1f}%")
    
    if gte_wins > e5_wins:
        print(f"\n🎉 GTE is the WINNER by {gte_wins - e5_wins} test(s)!")
    elif e5_wins > gte_wins:
        print(f"\n🎉 E5 is the WINNER by {e5_wins - gte_wins} test(s)!")
    else:
        print(f"\n🤝 It's a TIE!")
    
    print("="*80)


if __name__ == "__main__":
    run_comparison()
