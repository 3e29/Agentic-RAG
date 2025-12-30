"""
Test comparative query through full agent pipeline
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.query_analysis import query_analysis_agent
from src.agents.retrieval import retrieval_agent


def test_full_pipeline():
    # Comparative query (Arabic)
    query = "ما الفرق بين الصبر عند الشدة والشكر عند الرخاء في الأحاديث"
    
    print("=" * 70)
    print("FULL AGENT PIPELINE TEST - Comparative Analysis")
    print("=" * 70)
    print(f"Query: {query}")
    
    # Initial state
    state = {
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
    
    # Step 1: Query Analysis
    print("\n" + "-" * 70)
    print("STEP 1: QUERY ANALYSIS AGENT")
    print("-" * 70)
    state = query_analysis_agent(state)
    
    print(f"Intent: {state.get('query_intent')}")
    print(f"Language: {state.get('language')}")
    print(f"Sub-queries ({len(state.get('sub_queries') or [])} total):")
    for i, sq in enumerate(state.get("sub_queries") or []):
        print(f"  {i+1}. {sq}")
    
    # Step 2: Retrieval
    print("\n" + "-" * 70)
    print("STEP 2: RETRIEVAL AGENT")
    print("-" * 70)
    state = retrieval_agent(state)
    
    docs = state.get("retrieved_docs", [])
    print(f"Retrieved {len(docs)} documents")
    
    # Check for hadith 14591
    print("\n" + "-" * 70)
    print("RESULTS ANALYSIS")
    print("-" * 70)
    
    print("\nTop 10 Results:")
    found_14591 = False
    for i, doc in enumerate(docs[:10]):
        hadith_id = doc.get("hadith_id") if isinstance(doc, dict) else getattr(doc, "hadith_id", None)
        text = doc.get("text") if isinstance(doc, dict) else getattr(doc, "text", "")
        text_preview = text[:100].replace("\n", " ")
        
        marker = ""
        if hadith_id == 14591:
            marker = " *** TARGET HADITH ***"
            found_14591 = True
        
        print(f"  [{i+1}] Hadith #{hadith_id}{marker}")
        print(f"      {text_preview}...")
    
    if not found_14591:
        print("\n  ✗ Hadith 14591 (patience+gratitude) NOT in top 10")
    else:
        print("\n  ✓ Hadith 14591 (patience+gratitude) FOUND!")
    
    # Topic coverage
    print("\n" + "-" * 70)
    print("TOPIC COVERAGE")
    print("-" * 70)
    
    expected = ["patience", "صبر", "gratitude", "شكر", "شدة", "رخاء"]
    all_text = " ".join([
        (d.get("text") if isinstance(d, dict) else getattr(d, "text", str(d))).lower()
        for d in docs
    ])
    
    found_count = 0
    for topic in expected:
        found = topic.lower() in all_text
        if found:
            found_count += 1
        print(f"  {'✓' if found else '✗'} '{topic}'")
    
    print(f"\nFinal Score: {found_count}/{len(expected)}")
    
    return found_count, len(expected)


if __name__ == "__main__":
    test_full_pipeline()
