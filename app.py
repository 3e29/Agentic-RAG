"""
Hadith RAG System - Test UI

A simple Streamlit interface to test the full pipeline:
1. Query Analysis Agent
2. Retrieval Agent

Run with: streamlit run app.py
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables (including LangSmith config)
load_dotenv()

sys.path.insert(0, ".")

import streamlit as st
import time
from typing import Optional
from langsmith import traceable

# Import agents
from src.agents.query_analysis import query_analysis_agent
from src.agents.retrieval import retrieval_agent


# Page config
st.set_page_config(
    page_title="Hadith RAG System",
    page_icon="📚",
    layout="wide",
)

# Sidebar Status
with st.sidebar:
    st.header("System Status")
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        st.success("✅ LangSmith Tracing Active")
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        st.caption(f"Project: {project}")
    else:
        st.warning("⚠️ LangSmith Tracing Inactive")
        st.caption("Check .env file")

# Custom CSS
st.markdown("""
<style>
    .hadith-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .hadith-arabic {
        direction: rtl;
        text-align: right;
        font-size: 1.1em;
        line-height: 1.8;
    }
    .hadith-english {
        font-size: 1em;
        line-height: 1.6;
    }
    .metadata-tag {
        background-color: #e9ecef;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        margin-right: 5px;
    }
    .stage-header {
        background-color: #007bff;
        color: white;
        padding: 8px 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@traceable(name="full_pipeline_run")
def run_pipeline(query: str):
    """Run the full pipeline and return results."""
    results = {
        "query": query,
        "analysis": None,
        "documents": [],
        "metadata": {},
        "timings": {},
    }
    
    # Stage 1: Query Analysis
    start = time.time()
    state = {"original_query": query}
    
    try:
        analysis_result = query_analysis_agent(state)
        results["analysis"] = {
            "intent": analysis_result.get("query_intent", "unknown"),
            "corrected_query": analysis_result.get("corrected_query", query),
            "language": analysis_result.get("language", "unknown"),
            "desired_output_language": analysis_result.get("desired_output_language"),
            "target_collections": analysis_result.get("target_collections", []),
            "sub_queries": analysis_result.get("sub_queries"),
        }
        state.update(analysis_result)
        results["timings"]["analysis"] = time.time() - start
    except Exception as e:
        results["analysis"] = {"error": str(e)}
        results["timings"]["analysis"] = time.time() - start
        return results
    
    # Stage 2: Retrieval
    start = time.time()
    try:
        retrieval_result = retrieval_agent(state)
        results["documents"] = retrieval_result.get("retrieved_docs", [])
        results["metadata"] = retrieval_result.get("metadata", {})
        results["timings"]["retrieval"] = time.time() - start
    except Exception as e:
        results["metadata"]["error"] = str(e)
        results["timings"]["retrieval"] = time.time() - start
    
    return results


def display_analysis(analysis: dict):
    """Display query analysis results."""
    if "error" in analysis:
        st.error(f"Analysis Error: {analysis['error']}")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Intent", analysis.get("intent", "N/A"))
    with col2:
        st.metric("Language", analysis.get("language", "N/A"))
    with col3:
        st.metric("Output Language", analysis.get("desired_output_language", "Auto"))
    
    if analysis.get("corrected_query") != st.session_state.get("last_query"):
        st.info(f"📝 Corrected Query: {analysis.get('corrected_query')}")
    
    if analysis.get("target_collections"):
        st.write(f"🎯 Target Collections: {', '.join(analysis['target_collections'])}")
    
    if analysis.get("sub_queries") and len(analysis["sub_queries"]) > 1:
        st.write(f"🔀 Decomposed into {len(analysis['sub_queries'])} sub-queries")


def display_document(doc, index: int):
    """Display a single hadith document."""
    is_arabic = doc.language == "arabic"
    
    with st.container():
        # Header with metadata
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            st.write(f"**Hadith #{doc.hadith_id}**")
        with col2:
            collection = doc.collection or "Unknown"
            st.write(f"📚 {collection}")
        with col3:
            st.write(f"🌐 {doc.language}")
        with col4:
            if doc.score:
                st.write(f"⭐ {doc.score:.2f}")
        
        # Chapter info
        chapter_en = getattr(doc, 'chapter_title_en', None)
        chapter_ar = getattr(doc, 'chapter_title_ar', None)
        chapter = chapter_en or chapter_ar
        if chapter:
            st.caption(f"📖 Chapter: {chapter}")
        
        # Hadith text
        if is_arabic:
            st.markdown(f'<div class="hadith-arabic">{doc.text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="hadith-english">{doc.text}</div>', unsafe_allow_html=True)
        
        # Additional metadata
        with st.expander("📋 More Details"):
            meta_cols = st.columns(4)
            with meta_cols[0]:
                st.write(f"**Book:** {doc.book_id or 'N/A'}")
            with meta_cols[1]:
                st.write(f"**Chapter ID:** {doc.chapter_id or 'N/A'}")
            with meta_cols[2]:
                st.write(f"**Narrator:** {doc.narrator or 'N/A'}")
            with meta_cols[3]:
                st.write(f"**Chunks:** {doc.total_chunks or 1}")
        
        st.divider()


def main():
    st.title("📚 Hadith RAG System")
    st.caption("Test the full pipeline: Query Analysis → Retrieval")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This UI tests the complete Hadith RAG pipeline:
        
        1. **Query Analysis Agent**
           - Input source detection
           - Typo correction
           - Intent classification
           - Collection targeting
           - Query decomposition
        
        2. **Retrieval Agent**
           - Autonomous ReAct search
           - Hybrid search (semantic + BM25)
           - Cross-encoder reranking
           - Chunk reassembly
        """)
        
        st.divider()
        st.header("🔧 Settings")
        show_metadata = st.checkbox("Show retrieval metadata", value=False)
        show_timings = st.checkbox("Show timing details", value=True)
    
    # Main query input
    query = st.text_input(
        "🔍 Enter your query",
        placeholder="e.g., ما هو حديث النية؟ / What are the hadiths about patience?",
        key="query_input"
    )
    
    # Example queries
    st.caption("**Example queries:**")
    example_cols = st.columns(4)
    examples = [
        "ما هو أطول حديث في صحيح البخاري؟",
        "Hadiths about patience",
        "أحاديث عن الصبر بالإنجليزية",
        "Hadith number 3964 in Bukhari",
    ]
    
    for col, example in zip(example_cols, examples):
        if col.button(example[:25] + "..." if len(example) > 25 else example, key=example):
            st.session_state.query_input = example
            st.rerun()
    
    # Search button
    if st.button("🔎 Search", type="primary", use_container_width=True) or query:
        if query:
            st.session_state["last_query"] = query
            
            with st.spinner("Processing query..."):
                results = run_pipeline(query)
            
            # Display Analysis Results
            st.markdown('<div class="stage-header">📊 Stage 1: Query Analysis</div>', unsafe_allow_html=True)
            if results["analysis"]:
                display_analysis(results["analysis"])
            
            # Display Retrieval Results
            st.markdown('<div class="stage-header">📚 Stage 2: Retrieval Results</div>', unsafe_allow_html=True)
            
            documents = results.get("documents", [])
            
            if not documents:
                st.warning("No documents found for this query.")
            else:
                st.success(f"Found {len(documents)} relevant hadith(s)")
                
                for i, doc in enumerate(documents, 1):
                    display_document(doc, i)
            
            # Timing info
            if show_timings and results.get("timings"):
                st.divider()
                timing_cols = st.columns(3)
                timings = results["timings"]
                with timing_cols[0]:
                    st.metric("Analysis Time", f"{timings.get('analysis', 0)*1000:.0f}ms")
                with timing_cols[1]:
                    st.metric("Retrieval Time", f"{timings.get('retrieval', 0)*1000:.0f}ms")
                with timing_cols[2]:
                    total = sum(timings.values())
                    st.metric("Total Time", f"{total*1000:.0f}ms")
            
            # Metadata (optional)
            if show_metadata and results.get("metadata"):
                with st.expander("🔧 Retrieval Metadata"):
                    st.json(results["metadata"])
        else:
            st.info("Please enter a query to search.")


if __name__ == "__main__":
    main()
