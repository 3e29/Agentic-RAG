"""
Test: Additional Queries for Full Pipeline
1. Longest Hadith in Bukhari
2. Attributes of a Righteous Wife

This test uses the FULL PIPELINE:
1. Query Analysis Agent
2. Retrieval Agent
"""

import sys
sys.path.insert(0, ".")

from src.agents.query_analysis import query_analysis_agent
from src.agents.retrieval import retrieval_agent


def run_pipeline(query, show_details=True):
    """Run the full agent pipeline for a given query."""
    
    print("=" * 70)
    print(f"TEST QUERY: {query}")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: QUERY ANALYSIS AGENT
    # =========================================================================
    print("STEP 1: QUERY ANALYSIS AGENT")
    
    # Initial state
    state = {"original_query": query}
    
    # Run query analysis
    analysis_result = query_analysis_agent(state)
    
    # Extract results
    intent = analysis_result.get("query_intent", "unknown")
    sub_queries = analysis_result.get("sub_queries") or [query]
    corrected_query = analysis_result.get("corrected_query") or query
    desired_output_language = analysis_result.get("desired_output_language")
    
    print(f"  Intent: {intent}")
    print(f"  Corrected Query: {corrected_query}")
    print(f"  Desired Output Language: {desired_output_language}")
    if len(sub_queries) > 1:
        print(f"  Sub-queries: {len(sub_queries)}")
    
    # Update state for retrieval
    state.update(analysis_result)
    
    # =========================================================================
    # STEP 2: RETRIEVAL AGENT
    # =========================================================================
    print("\nSTEP 2: RETRIEVAL AGENT")
    
    # Run retrieval agent
    retrieval_result = retrieval_agent(state)
    
    documents = retrieval_result.get("retrieved_docs", [])
    metadata = retrieval_result.get("metadata", {})
    
    print(f"  Retrieved {len(documents)} documents")
    
    # Show agent iterations if available
    retrieval_meta = metadata.get("retrieval", {})
    agent_iterations = retrieval_meta.get("agent_iterations", [])
    if agent_iterations:
        print("  Agent Iterations:")
        for it in agent_iterations:
            print(f"    [{it.get('query_index', 0)}] {it.get('action', '?')} -> {it.get('result', '')[:50]}...")

    return documents


def test_longest_hadith():
    """Test 1: Longest Hadith in Bukhari."""
    query = "ماهو اطول حديث في صحيح البخاري؟"
    documents = run_pipeline(query)
    
    print("\n" + "-" * 70)
    print("RESULTS (Longest Hadith)")
    print("-" * 70)
    
    if not documents:
        print("No documents found.")
    else:
        doc = documents[0]
        print(f"Hadith #{doc.hadith_id}")
        print(f"Collection: {doc.collection}")
        print(f"Language: {doc.language}")
        chapter_en = getattr(doc, 'chapter_title_en', None)
        chapter_ar = getattr(doc, 'chapter_title_ar', None)
        print(f"Chapter: {chapter_en or chapter_ar or 'N/A'}")
        print(f"\nFull Text:\n{doc.text}")
    print("\n")


def test_righteous_wife():
    """Test 2: Attributes of a Righteous Wife."""
    query = "صفات الزوجة الصالحة"
    documents = run_pipeline(query)
    
    print("\n" + "-" * 70)
    print("RESULTS (Details)")
    print("-" * 70)
    
    if not documents:
        print("No documents found.")
    else:
        for i, doc in enumerate(documents, 1):
            print(f"[{i}] Hadith #{doc.hadith_id}")
            print(f"    Collection: {doc.collection}")
            print(f"    Language: {doc.language}")
            # Use getattr for chapter titles as they may not exist on all documents
            chapter_en = getattr(doc, 'chapter_title_en', None)
            chapter_ar = getattr(doc, 'chapter_title_ar', None)
            print(f"    Chapter: {chapter_en or chapter_ar or 'N/A'}")
            # Show text preview
            text = doc.text or ""
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"    Text: {text}")
            print()


def test_language_preference():
    """Test 3: Language preference detection."""
    query = "أريد أحاديث عن الصبر بالإنجليزية"
    documents = run_pipeline(query)
    
    print("\n" + "-" * 70)
    print("RESULTS (Details)")
    print("-" * 70)
    
    if not documents:
        print("No documents found.")
    else:
        for i, doc in enumerate(documents[:3], 1):
            print(f"[{i}] Hadith #{doc.hadith_id}")
            print(f"    Collection: {doc.collection}")
            print(f"    Language: {doc.language}")
            text = doc.text or ""
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"    Text: {text}")
            print()


if __name__ == "__main__":
    test_longest_hadith()
    test_righteous_wife()
    test_language_preference()
