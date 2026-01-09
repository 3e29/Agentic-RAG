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
from typing import Optional, List, Dict, Any
from langsmith import traceable

# Import agents
from src.agents.query_analysis import query_analysis_agent
from src.agents.retrieval import retrieval_agent


# Page config
st.set_page_config(
    page_title="Hadith RAG System",
    layout="wide",
)


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


def build_execution_graph(results: Dict[str, Any]) -> str:
    """
    Build a Graphviz DOT string based on the execution flow.
    
    Args:
        results: Pipeline results containing analysis and metadata
        
    Returns:
        DOT format string for graphviz rendering
    """
    # Extract execution info
    analysis = results.get("analysis", {})
    metadata = results.get("metadata", {})
    retrieval_meta = metadata.get("retrieval", {})
    evaluation_meta = metadata.get("evaluation", {})
    
    query_intent = analysis.get("intent", "unknown")
    stages = retrieval_meta.get("stages", [])
    iterations = retrieval_meta.get("agent_iterations", [])
    
    # Evaluation data
    eval_status = evaluation_meta.get("status", "STOP")
    eval_iteration = evaluation_meta.get("iteration", 0)
    confidence_score = evaluation_meta.get("confidence_score", 0.0)
    quality_score = evaluation_meta.get("quality_score", 0.0)
    grounding_score = evaluation_meta.get("grounding_score", 0.0)
    coverage_score = evaluation_meta.get("coverage_score", 0.0)
    
    # Node styles
    executed_style = 'style="filled" fillcolor="#90EE90" color="#228B22"'  # Green
    skipped_style = 'style="filled" fillcolor="#D3D3D3" color="#808080"'   # Gray
    current_style = 'style="filled" fillcolor="#FFD700" color="#FF8C00"'   # Gold
    continue_style = 'style="filled" fillcolor="#FFB347" color="#FF8C00"'  # Orange for continue
    stop_style = 'style="filled" fillcolor="#32CD32" color="#228B22"'      # Bright green for stop
    
    # Determine evaluation style based on status
    eval_decision_style = stop_style if eval_status == "STOP" else continue_style
    
    # Build DOT graph
    dot = '''
    digraph LangGraph {
        rankdir=TB;
        node [shape=box, fontname="Arial", fontsize=10];
        edge [fontname="Arial", fontsize=9];
        
        // Start node
        start [label="Start" shape=circle style="filled" fillcolor="#4169E1" fontcolor="white"];
        
        // Query Analysis Stage
        subgraph cluster_analysis {
            label="Query Analysis Agent";
            style=rounded;
            bgcolor="#E6F3FF";
            
            qa_input [label="Input Query" ''' + executed_style + '''];
            qa_intent [label="Intent Detection\\n(''' + query_intent + ''')" ''' + executed_style + '''];
            qa_lang [label="Language Detection" ''' + executed_style + '''];
            qa_correct [label="Query Correction" ''' + executed_style + '''];
            qa_decompose [label="Query Decomposition" ''' + executed_style + '''];
        }
        
        // Retrieval Stage
        subgraph cluster_retrieval {
            label="Retrieval Agent";
            style=rounded;
            bgcolor="#FFF3E6";
    '''
    
    # Determine which retrieval path was taken
    is_metadata_query = "metadata_query" in stages
    is_user_text = "user_text_processing" in stages
    is_autonomous = not is_metadata_query and not is_user_text
    
    # Metadata query path
    if is_metadata_query:
        dot += '''
            ret_router [label="Intent Router" ''' + executed_style + '''];
            ret_metadata [label="Metadata Query\\n(longest/shortest)" ''' + executed_style + '''];
            ret_extract [label="Extract Filters\\n(narrator, chapter)" ''' + executed_style + '''];
            ret_resolve [label="Resolve Chapter ID" ''' + executed_style + '''];
            ret_repo [label="Repository Query" ''' + executed_style + '''];
        '''
    elif is_user_text:
        dot += '''
            ret_router [label="Intent Router" ''' + executed_style + '''];
            ret_user [label="User Text\\nProcessor" ''' + executed_style + '''];
            ret_similarity [label="Find Similar\\nHadiths" ''' + executed_style + '''];
        '''
    else:
        # Autonomous search - show iterations
        dot += '''
            ret_router [label="Intent Router" ''' + executed_style + '''];
            ret_orchestrator [label="Search\\nOrchestrator" ''' + executed_style + '''];
        '''
        
        # Add iteration nodes based on actual execution
        if iterations:
            for i, iteration in enumerate(iterations[:5]):  # Limit to 5 iterations
                action = iteration.get("action", "unknown")
                action_labels = {
                    "expand_query": "Expand Query",
                    "extract_filters": "Extract Filters",
                    "find_chapter": "Find Chapter",
                    "keyword_search": "Keyword Search",
                    "semantic_search": "Semantic Search",
                    "hybrid_search": "Hybrid Search",
                    "relax_filters": "Relax Filters",
                    "finish": "Finish",
                }
                label = action_labels.get(action, action)
                dot += f'''
            iter_{i} [label="{label}" {executed_style}];
                '''
        else:
            # Default autonomous flow
            dot += '''
            ret_hybrid [label="Hybrid Search" ''' + executed_style + '''];
            '''
    
    # Aggregation and output
    dot += '''
            ret_aggregate [label="Aggregate\\n& Rerank" ''' + executed_style + '''];
            ret_reassemble [label="Reassemble\\nChunks" ''' + executed_style + '''];
        }
        
        // Evaluation Agent Stage
        subgraph cluster_evaluation {
            label="Evaluation Agent";
            style=rounded;
            bgcolor="#E8F5E9";
            
            eval_quality [label="Quality\\nAssessment" ''' + executed_style + '''];
            eval_gaps [label="Gap\\nIdentification" ''' + executed_style + '''];
            eval_grounding [label="Grounding\\nValidation" ''' + executed_style + '''];
            eval_decision [label="''' + ("STOP" if eval_status == "STOP" else "CONTINUE") + '''\\n(confidence: ''' + f"{confidence_score:.2f}" + ''')" ''' + eval_decision_style + '''];
        }
        
        // End node
        end [label="Results" shape=circle style="filled" fillcolor="#228B22" fontcolor="white"];
        
        // Edges - Analysis flow
        start -> qa_input;
        qa_input -> qa_intent;
        qa_intent -> qa_lang;
        qa_lang -> qa_correct;
        qa_correct -> qa_decompose;
        qa_decompose -> ret_router;
    '''
    
    # Edges - Retrieval flow based on path
    if is_metadata_query:
        dot += '''
        ret_router -> ret_metadata [label="metadata_query"];
        ret_metadata -> ret_extract;
        ret_extract -> ret_resolve;
        ret_resolve -> ret_repo;
        ret_repo -> ret_aggregate;
        '''
    elif is_user_text:
        dot += '''
        ret_router -> ret_user [label="user_text"];
        ret_user -> ret_similarity;
        ret_similarity -> ret_aggregate;
        '''
    else:
        dot += '''
        ret_router -> ret_orchestrator [label="autonomous"];
        '''
        if iterations:
            # Chain iterations
            dot += f'ret_orchestrator -> iter_0;\n'
            for i in range(len(iterations[:5]) - 1):
                dot += f'        iter_{i} -> iter_{i+1};\n'
            dot += f'        iter_{len(iterations[:5])-1} -> ret_aggregate;\n'
        else:
            dot += '''
        ret_orchestrator -> ret_hybrid;
        ret_hybrid -> ret_aggregate;
            '''
    
    # Connect retrieval to evaluation
    dot += '''
        ret_aggregate -> ret_reassemble;
        ret_reassemble -> eval_quality;
        
        // Evaluation flow
        eval_quality -> eval_gaps;
        eval_gaps -> eval_grounding;
        eval_grounding -> eval_decision;
    '''
    
    # Connect evaluation to end or back to retrieval (if continue)
    if eval_status == "CONTINUE" and eval_iteration < 3:
        dot += '''
        eval_decision -> ret_router [label="retry" style="dashed" color="#FF8C00"];
        '''
    
    dot += '''
        eval_decision -> end;
    }
    '''
    
    return dot


def render_sidebar_graph(results: Optional[Dict[str, Any]] = None):
    """Render the execution graph in the sidebar."""
    st.sidebar.title("🔄 Execution Flow")
    
    if results is None:
        # Show default/idle graph
        default_dot = '''
        digraph LangGraph {
            rankdir=TB;
            node [shape=box, fontname="Arial", fontsize=10];
            
            start [label="Start" shape=circle style="filled" fillcolor="#4169E1" fontcolor="white"];
            qa [label="Query Analysis\\nAgent" style="filled" fillcolor="#E6F3FF"];
            ret [label="Retrieval\\nAgent" style="filled" fillcolor="#FFF3E6"];
            eval [label="Evaluation\\nAgent" style="filled" fillcolor="#E8F5E9"];
            end [label="Results" shape=circle style="filled" fillcolor="#D3D3D3"];
            
            start -> qa [label="query"];
            qa -> ret [label="state"];
            ret -> eval [label="docs"];
            eval -> ret [label="continue" style="dashed" color="#FF8C00"];
            eval -> end [label="stop"];
            
            note [label="Enter a query to see\\nthe execution flow" shape=note style="filled" fillcolor="#FFFACD"];
        }
        '''
        st.sidebar.graphviz_chart(default_dot)
        
        # Legend
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Legend:**")
        st.sidebar.markdown("Executed node")
        st.sidebar.markdown("Continue (retry)")
        st.sidebar.markdown("Skipped node")
    else:
        # Build and render execution graph
        dot = build_execution_graph(results)
        st.sidebar.graphviz_chart(dot)
        
        # Show execution stats
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Execution Stats:**")
        
        metadata = results.get("metadata", {})
        retrieval_meta = metadata.get("retrieval", {})
        evaluation_meta = metadata.get("evaluation", {})
        
        stages = retrieval_meta.get("stages", [])
        iterations = retrieval_meta.get("agent_iterations", [])
        
        st.sidebar.metric("Stages", len(stages))
        st.sidebar.metric("Agent Iterations", len(iterations))
        
        if results.get("timings"):
            total_time = sum(results["timings"].values()) * 1000
            st.sidebar.metric("Total Time", f"{total_time:.0f}ms")
        
        # Show evaluation metrics
        if evaluation_meta:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Evaluation:**")
            
            eval_status = evaluation_meta.get("status", "N/A")
            confidence = evaluation_meta.get("confidence_score", 0.0)
            quality = evaluation_meta.get("quality_score", 0.0)
            grounding = evaluation_meta.get("grounding_score", 0.0)
            coverage = evaluation_meta.get("coverage_score", 0.0)
            
            status_color = "" if eval_status == "STOP" else ""
            st.sidebar.markdown(f"**Status:** {status_color} {eval_status}")
            
            # Show iteration count
            eval_iteration = evaluation_meta.get("iteration", 1)
            st.sidebar.markdown(f"**Iterations:** {eval_iteration}")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Confidence", f"{confidence:.2f}" if confidence else "N/A")
                st.metric("Quality", f"{quality:.2f}" if quality else "N/A")
            with col2:
                st.metric("Grounding", f"{grounding:.2f}" if grounding else "N/A")
                st.metric("Coverage", f"{coverage:.2f}" if coverage else "N/A")
        
        # Show stages list
        if stages:
            st.sidebar.markdown("**Stages Executed:**")
            for stage in stages:
                st.sidebar.markdown(f"• {stage}")


@traceable(name="full_pipeline_run")
def run_pipeline(query: str):
    """Run the full pipeline using LangGraph workflow with evaluation loop."""
    from src.graph.workflow import create_workflow
    
    results = {
        "query": query,
        "analysis": None,
        "documents": [],
        "metadata": {},
        "timings": {},
    }
    
    start_total = time.time()
    
    try:
        # Create workflow with evaluation enabled
        workflow = create_workflow(enable_evaluation=True)
        
        # Initialize state
        initial_state = {
            "original_query": query,
            "normalized_query": None,
            "corrected_query": None,
            "search_query": None,
            "input_source": None,
            "query_intent": None,
            "target_collections": None,
            "sub_queries": None,
            "search_sub_queries": None,  # Optimized sub-queries for embedding
            "retrieved_docs": None,
            "evaluation_feedback": None,
            "confidence_score": None,
            "missing_information_gaps": None,
            "language": None,
            "desired_output_language": None,
            "metadata": {},
        }
        
        # Execute workflow (includes evaluation loop)
        final_state = workflow.invoke(initial_state)
        
        # Extract results from final state
        results["analysis"] = {
            "intent": final_state.get("query_intent", "unknown"),
            "corrected_query": final_state.get("corrected_query", query),
            "search_query": final_state.get("search_query"),
            "language": final_state.get("language", "unknown"),
            "desired_output_language": final_state.get("desired_output_language"),
            "target_collections": final_state.get("target_collections", []),
            "sub_queries": final_state.get("sub_queries"),
            "search_sub_queries": final_state.get("search_sub_queries"),  # Add this too
        }
        
        results["documents"] = final_state.get("retrieved_docs", [])
        results["metadata"] = final_state.get("metadata", {})
        results["evaluation_feedback"] = final_state.get("evaluation_feedback")
        results["confidence_score"] = final_state.get("confidence_score")
        
        # Calculate total time
        results["timings"]["total"] = time.time() - start_total
        
        # Extract individual timings from metadata if available
        query_analysis_meta = results["metadata"].get("query_analysis", {})
        retrieval_meta = results["metadata"].get("retrieval", {})
        evaluation_meta = results["metadata"].get("evaluation", {})
        
        # Rough time estimates from metadata
        if retrieval_meta.get("total_execution_time_ms"):
            results["timings"]["retrieval"] = retrieval_meta["total_execution_time_ms"] / 1000
        if evaluation_meta.get("execution_time_ms"):
            results["timings"]["evaluation"] = evaluation_meta["execution_time_ms"] / 1000
        
        # Analysis time = total - retrieval - evaluation (approximate)
        retrieval_time = results["timings"].get("retrieval", 0)
        eval_time = results["timings"].get("evaluation", 0)
        results["timings"]["analysis"] = max(0, results["timings"]["total"] - retrieval_time - eval_time)
        
    except Exception as e:
        results["metadata"]["error"] = str(e)
        results["timings"]["total"] = time.time() - start_total
        import traceback
        traceback.print_exc()
    
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
        st.info(f"Corrected Query: {analysis.get('corrected_query')}")
    
    # Show search query (optimized for embedding)
    search_query = analysis.get("search_query")
    if search_query and search_query != analysis.get("corrected_query"):
        st.success(f"Search Query (for embedding): {search_query}")
    
    if analysis.get("target_collections"):
        st.write(f"Target Collections: {', '.join(analysis['target_collections'])}")
    
    if analysis.get("sub_queries") and len(analysis["sub_queries"]) > 1:
        st.write(f"Decomposed into {len(analysis['sub_queries'])} sub-queries")


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
            st.write(f"{collection}")
        with col3:
            st.write(f"{doc.language}")
        with col4:
            if doc.score:
                st.write(f"{doc.score:.2f}")
        
        # Chapter info
        chapter_en = getattr(doc, 'chapter_title_en', None)
        chapter_ar = getattr(doc, 'chapter_title_ar', None)
        chapter = chapter_en or chapter_ar
        if chapter:
            st.caption(f"Chapter: {chapter}")
        
        # Hadith text
        if is_arabic:
            st.markdown(f'<div class="hadith-arabic">{doc.text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="hadith-english">{doc.text}</div>', unsafe_allow_html=True)
        
        # Additional metadata
        with st.expander("More Details"):
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
    st.title("Hadith RAG System")
    st.caption("Test the full pipeline: Query Analysis → Retrieval")
    
    # Initialize session state for results
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None
    
    # Render sidebar graph (updates based on results)
    render_sidebar_graph(st.session_state.pipeline_results)
    
    # Sidebar options
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Options:**")
    show_timings = st.sidebar.checkbox("Show Timings", value=True)
    show_metadata = st.sidebar.checkbox("Show Metadata", value=False)
    
    # Main query input
    query = st.text_input(
        "Enter your query",
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
            st.session_state["run_search"] = True
            st.rerun()
    
    # Search button - only run when button is clicked (not on every rerun)
    search_clicked = st.button("Search", type="primary", use_container_width=True)
    
    # Check if we should run a search (button clicked or flagged from example)
    should_search = search_clicked or st.session_state.get("run_search", False)
    
    if should_search and query:
        # Clear the flag
        st.session_state["run_search"] = False
        
        # Only run if query changed or explicit search requested
        if query != st.session_state.get("last_query") or search_clicked:
            st.session_state["last_query"] = query
            
            with st.spinner("Processing query..."):
                results = run_pipeline(query)
            
            # Store results for sidebar graph
            st.session_state.pipeline_results = results
            
            # Force rerun to update sidebar graph
            st.rerun()
    
    # Display results if available
    if st.session_state.pipeline_results and st.session_state.get("last_query"):
        results = st.session_state.pipeline_results
        
        # Display Analysis Results
        st.markdown('<div class="stage-header">Stage 1: Query Analysis</div>', unsafe_allow_html=True)
        if results["analysis"]:
            display_analysis(results["analysis"])
        
        # Display Retrieval Results
        st.markdown('<div class="stage-header">Stage 2: Retrieval Results</div>', unsafe_allow_html=True)
        
        documents = results.get("documents", [])
        
        if not documents:
            st.warning("No documents found for this query.")
        else:
            st.success(f"Found {len(documents)} relevant hadith(s)")
            
            for i, doc in enumerate(documents, 1):
                display_document(doc, i)
        
        # Display Evaluation Results
        if results.get("evaluation_feedback"):
            st.markdown('<div class="stage-header">Stage 3: Evaluation Results</div>', unsafe_allow_html=True)
            eval_meta = results.get("metadata", {}).get("evaluation", {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status = eval_meta.get("status", "N/A")
                status_icon = "" if status == "STOP" else ""
                st.metric("Status", f"{status_icon} {status}")
            with col2:
                st.metric("Confidence", f"{results.get('confidence_score', 0):.2f}")
            with col3:
                st.metric("Quality", f"{eval_meta.get('quality_score', 0) or 0:.2f}")
            with col4:
                st.metric("Grounding", f"{eval_meta.get('grounding_score', 0) or 0:.2f}")
            
            if results.get("evaluation_feedback"):
                st.info(f"**Feedback:** {results['evaluation_feedback']}")
        
        # Timing info
        if show_timings and results.get("timings"):
            st.divider()
            timing_cols = st.columns(4)
            timings = results["timings"]
            with timing_cols[0]:
                st.metric("Analysis Time", f"{timings.get('analysis', 0)*1000:.0f}ms")
            with timing_cols[1]:
                st.metric("Retrieval Time", f"{timings.get('retrieval', 0)*1000:.0f}ms")
            with timing_cols[2]:
                st.metric("Evaluation Time", f"{timings.get('evaluation', 0)*1000:.0f}ms")
            with timing_cols[3]:
                total = sum(timings.values())
                st.metric("Total Time", f"{total*1000:.0f}ms")
        
        # Metadata (optional)
        if show_metadata and results.get("metadata"):
            with st.expander("Full Pipeline Metadata"):
                st.json(results["metadata"])
    elif not st.session_state.get("last_query"):
        st.info("Please enter a query to search.")


if __name__ == "__main__":
    main()
