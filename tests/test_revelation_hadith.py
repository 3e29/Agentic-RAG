"""
Test: Search for Hadith about Beginning of Revelation
Query: ابحث عن حديث بدء الوحي وكيف نزل جبريل على النبي في غار حراء
(Search for the hadith about the beginning of revelation and how Gabriel descended upon the Prophet in the Cave of Hira)

This test uses the FULL PIPELINE:
1. Query Analysis Agent - intent classification, decomposition
2. Retrieval Agent - search with autonomous agent
"""

import asyncio
import sys
sys.path.insert(0, ".")

from src.agents.query_analysis import query_analysis_agent
from src.agents.retrieval import retrieval_agent


def run_full_pipeline_test():
    """Run the full agent pipeline for revelation hadith query."""
    
    query = "ابحث عن حديث بدء الوحي وكيف نزل جبريل على النبي في غار حراء"
    
    print("=" * 70)
    print("FULL PIPELINE TEST - Beginning of Revelation Hadith")
    print("=" * 70)
    print(f"Query: {query}")
    print()
    
    # =========================================================================
    # STEP 1: QUERY ANALYSIS AGENT
    # =========================================================================
    print("-" * 70)
    print("STEP 1: QUERY ANALYSIS AGENT")
    print("-" * 70)
    
    # Initial state
    state = {"original_query": query}
    
    # Run query analysis
    analysis_result = query_analysis_agent(state)
    
    # Extract results
    intent = analysis_result.get("intent", "unknown")
    language = analysis_result.get("language", "unknown")
    sub_queries = analysis_result.get("sub_queries") or [query]
    corrected_query = analysis_result.get("corrected_query") or query
    
    print(f"Intent: {intent}")
    print(f"Language: {language}")
    print(f"Corrected Query: {corrected_query}")
    print(f"Sub-queries ({len(sub_queries)} total):")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")
    
    # Update state for retrieval
    state.update(analysis_result)
    
    # =========================================================================
    # STEP 2: RETRIEVAL AGENT
    # =========================================================================
    print()
    print("-" * 70)
    print("STEP 2: RETRIEVAL AGENT")
    print("-" * 70)
    
    # Run retrieval agent
    retrieval_result = retrieval_agent(state)
    
    documents = retrieval_result.get("retrieved_docs", [])
    metadata = retrieval_result.get("metadata", {})
    
    print(f"Retrieved {len(documents)} documents")
    
    # Show agent iterations if available
    retrieval_meta = metadata.get("retrieval", {})
    agent_iterations = retrieval_meta.get("agent_iterations", [])
    if agent_iterations:
        print()
        print("Agent Iterations:")
        for it in agent_iterations:
            print(f"  [{it.get('query_index', 0)}] Iteration {it.get('iteration', '?')}: {it.get('action', '?')}")
            thought = it.get('thought', '')
            if thought:
                print(f"      Thought: {thought[:80]}...")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print()
    print("-" * 70)
    print("RETRIEVED HADITHS")
    print("-" * 70)
    
    for i, doc in enumerate(documents, 1):
        print(f"\n[{i}] Hadith #{doc.hadith_id} ({doc.collection})")
        
        # Get chapter title from metadata if available
        chapter = getattr(doc, 'chapter_title_en', None) or getattr(doc, 'chapter_title_ar', None) or 'N/A'
        print(f"    Chapter: {chapter}")
        
        # Show text preview
        text = doc.text or ""
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"    Text: {text}")
    
    # =========================================================================
    # KEY TERMS CHECK
    # =========================================================================
    print()
    print("-" * 70)
    print("KEY TERMS COVERAGE")
    print("-" * 70)
    
    key_terms = [
        ("وحي", "revelation"),
        ("جبريل", "Gabriel"),
        ("حراء", "Hira"),
        ("غار", "cave"),
        ("اقرأ", "Read/Iqra"),
        ("خديجة", "Khadijah"),
        ("ورقة", "Waraqa"),
    ]
    
    all_text = " ".join([
        doc.text or ""
        for doc in documents
    ]).lower()
    
    found_count = 0
    for ar_term, en_term in key_terms:
        found = ar_term in all_text or en_term.lower() in all_text
        status = "✓" if found else "✗"
        if found:
            found_count += 1
        print(f"  {status} '{ar_term}' / '{en_term}'")
    
    print()
    print(f"Topic Coverage: {found_count}/{len(key_terms)}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Query: {query}")
    print(f"  Intent: {intent}")
    print(f"  Documents Retrieved: {len(documents)}")
    print(f"  Topic Coverage: {found_count}/{len(key_terms)}")
    
    # Check for famous hadith #1 (first hadith in Bukhari about revelation)
    hadith_ids = [doc.hadith_id for doc in documents]
    if 1 in hadith_ids or 3 in hadith_ids or 4 in hadith_ids or 6 in hadith_ids:
        print("  ✓ Found hadiths from Book of Revelation!")
    
    print("=" * 70)
    
    return documents


if __name__ == "__main__":
    docs = run_full_pipeline_test()
