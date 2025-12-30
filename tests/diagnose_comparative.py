"""
Diagnostic script for Comparative Analysis test performance
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.retrieval.search_tools import hybrid_search, semantic_search, keyword_search

def diagnose_comparative():
    # Test 3A: English comparative query
    query_en = "What is the difference between patience during hardship and gratitude during ease according to hadith?"
    query_ar = "ما الفرق بين الصبر عند الشدة والشكر عند الرخاء في الأحاديث"
    
    print("=" * 80)
    print("DIAGNOSTIC: Comparative Analysis Performance")
    print("=" * 80)
    
    # =========================================================================
    # ENGLISH QUERY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: ENGLISH QUERY")
    print("=" * 80)
    print(f"Query: {query_en}\n")
    
    # Semantic search
    print("1. SEMANTIC SEARCH (top 5):")
    sem_result = semantic_search(query_en, k=10)
    for i, doc in enumerate(sem_result.documents[:5]):
        text_preview = doc.text[:120].replace('\n', ' ')
        print(f"   [{i+1}] Score: {doc.score:.4f}")
        print(f"       {text_preview}...")
    
    # Keyword search
    print("\n2. BM25 KEYWORD SEARCH (top 5):")
    kw_result = keyword_search(query_en, k=10)
    for i, doc in enumerate(kw_result.documents[:5]):
        text_preview = doc.text[:120].replace('\n', ' ')
        print(f"   [{i+1}] Score: {doc.score:.4f}")
        print(f"       {text_preview}...")
    
    # Topic coverage
    print("\n3. TOPIC COVERAGE IN SEMANTIC RESULTS:")
    expected_en = ["patience", "صبر", "gratitude", "شكر", "hardship", "ease"]
    all_text_sem = " ".join([doc.text.lower() for doc in sem_result.documents])
    for topic in expected_en:
        found = topic.lower() in all_text_sem
        print(f"   {'✓' if found else '✗'} '{topic}' - {'FOUND' if found else 'NOT FOUND'}")
    
    # =========================================================================
    # ARABIC QUERY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: ARABIC QUERY")
    print("=" * 80)
    print(f"Query: {query_ar}\n")
    
    # Semantic search for Arabic
    print("1. SEMANTIC SEARCH (top 5):")
    sem_ar = semantic_search(query_ar, k=10)
    for i, doc in enumerate(sem_ar.documents[:5]):
        text_preview = doc.text[:120].replace('\n', ' ')
        print(f"   [{i+1}] Score: {doc.score:.4f}")
        print(f"       {text_preview}...")
    
    # Topic coverage
    print("\n2. TOPIC COVERAGE IN ARABIC SEMANTIC RESULTS:")
    expected_ar = ["patience", "صبر", "gratitude", "شكر", "شدة", "رخاء"]
    all_text_ar = " ".join([doc.text.lower() for doc in sem_ar.documents])
    for topic in expected_ar:
        found = topic.lower() in all_text_ar
        print(f"   {'✓' if found else '✗'} '{topic}' - {'FOUND' if found else 'NOT FOUND'}")
    
    # =========================================================================
    # ROOT CAUSE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    # Search for patience specifically
    print("\n1. DIRECT SEARCH FOR 'صبر' (patience):")
    patience_result = semantic_search("الصبر patience", k=5)
    for i, doc in enumerate(patience_result.documents[:3]):
        text_preview = doc.text[:150].replace('\n', ' ')
        print(f"   [{i+1}] {text_preview}...")
    
    # Search for gratitude specifically
    print("\n2. DIRECT SEARCH FOR 'شكر' (gratitude):")
    gratitude_result = semantic_search("الشكر gratitude thankfulness", k=5)
    for i, doc in enumerate(gratitude_result.documents[:3]):
        text_preview = doc.text[:150].replace('\n', ' ')
        print(f"   [{i+1}] {text_preview}...")
    
    # Check if there are hadiths about both together
    print("\n3. SEARCH FOR PATIENCE + GRATITUDE TOGETHER:")
    both_result = semantic_search("الصبر والشكر patience and gratitude believer", k=5)
    for i, doc in enumerate(both_result.documents[:3]):
        text_preview = doc.text[:150].replace('\n', ' ')
        has_patience = "صبر" in doc.text or "patience" in doc.text.lower()
        has_gratitude = "شكر" in doc.text or "gratitude" in doc.text.lower()
        print(f"   [{i+1}] patience={has_patience}, gratitude={has_gratitude}")
        print(f"       {text_preview}...")


if __name__ == "__main__":
    diagnose_comparative()
