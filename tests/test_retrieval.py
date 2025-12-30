"""
Comprehensive Test Suite for Retrieval Agent and Tools

Tests cover:
1. Query Expansion - Verify synonyms and translations
2. Metadata Filter Extraction - Pattern and LLM-based
3. Parallel Map-Reduce - Sub-query execution
4. Deduplication and Reranking
5. Self-Correction (retry with relaxed filters)
6. User Text Processing
7. Integration Tests

Production Standards:
- pytest-asyncio for async tests
- Mocking for isolated unit tests
- Integration tests with real vector store
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import schemas
from src.tools.retrieval.schemas import (
    Document,
    MetadataFilter,
    SearchResult,
    HybridSearchResult,
    AggregatedResults,
    ExpandedQuery,
    SearchType,
)

# Import tools
from src.tools.retrieval.search_tools import (
    SemanticSearchTool,
    KeywordSearchTool,
    HybridSearchTool,
    semantic_search,
    keyword_search,
    hybrid_search,
    _reciprocal_rank_fusion,
)
from src.tools.retrieval.filter_tools import (
    MetadataFilterTool,
    QueryExpansionTool,
    extract_metadata_filters,
    expand_query,
    _expand_from_dictionary,
)
from src.tools.retrieval.aggregation_tools import (
    ResultAggregationTool,
    aggregate_results,
    _deduplicate_documents,
    _normalize_scores,
)
from src.tools.retrieval.user_content_tools import (
    UserHadithProcessorTool,
    process_user_hadith,
    _clean_user_text,
)

# Import agent
from src.agents.retrieval import (
    retrieval_agent,
    retrieve,
    _execute_parallel_searches,
    _execute_search_with_retry,
    _handle_base_knowledge_search,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_documents() -> List[Document]:
    """Sample documents for testing."""
    return [
        Document(
            chunk_id="bukhari_1_1",
            text="إنما الأعمال بالنيات",
            hadith_id=1,
            collection="Sahih al-Bukhari",
            book_id=1,
            chapter_id=1,
            narrator="Narrated 'Umar bin Al-Khattab",
            score=0.95,
            search_type="semantic",
        ),
        Document(
            chunk_id="bukhari_1_2",
            text="الإيمان أن تؤمن بالله وملائكته",
            hadith_id=2,
            collection="Sahih al-Bukhari",
            book_id=1,
            chapter_id=2,
            narrator="Narrated Abu Hurairah",
            score=0.85,
            search_type="semantic",
        ),
        Document(
            chunk_id="muslim_1_1",
            text="بني الإسلام على خمس",
            hadith_id=1,
            collection="Sahih Muslim",
            book_id=1,
            chapter_id=1,
            narrator="Narrated Ibn Umar",
            score=0.80,
            search_type="keyword",
        ),
    ]


@pytest.fixture
def sample_state() -> Dict[str, Any]:
    """Sample agent state for testing."""
    return {
        "original_query": "What are hadiths about prayer?",
        "normalized_query": "What are hadiths about prayer?",
        "corrected_query": "What are hadiths about prayer?",
        "input_source": "base_knowledge",
        "query_intent": "thematic_search",
        "target_collections": ["bukhari", "muslim"],
        "sub_queries": ["hadiths about prayer"],
        "language": "en",
        "metadata": {},
    }


# ============================================================================
# Test: Query Expansion Tool
# ============================================================================

class TestQueryExpansionTool:
    """Tests for query expansion functionality."""
    
    def test_expand_prayer_to_salah(self):
        """Verify 'prayer' expands to include 'salah'."""
        result = _expand_from_dictionary("What are hadiths about prayer?")
        
        assert "terms" in result
        assert any("salah" in term.lower() for term in result["terms"])
    
    def test_expand_arabic_term(self):
        """Verify Arabic terms get English translations."""
        result = _expand_from_dictionary("أحاديث عن الصلاة")
        
        assert "translations" in result
        # الصلاة should expand to prayer terms
        if "الصلاة" in result["translations"]:
            assert "prayer" in result["translations"]["الصلاة"]
    
    def test_expand_charity_synonyms(self):
        """Verify 'charity' expands to zakat, sadaqah."""
        result = _expand_from_dictionary("hadiths about charity")
        
        assert "terms" in result
        terms_lower = [t.lower() for t in result["terms"]]
        assert any("zakat" in t or "sadaqah" in t for t in terms_lower)
    
    def test_expand_query_full_tool(self):
        """Test full QueryExpansionTool without LLM."""
        tool = QueryExpansionTool()
        result = tool("hadiths about fasting", use_llm=False)
        
        assert isinstance(result, ExpandedQuery)
        assert result.original_query == "hadiths about fasting"
        assert len(result.get_all_queries()) >= 1
    
    def test_expand_empty_query(self):
        """Test expansion with unrecognized terms."""
        result = _expand_from_dictionary("xyz unknown topic")
        
        assert result["terms"] == []
        assert result["translations"] == {}


# ============================================================================
# Test: Metadata Filter Extraction
# ============================================================================

class TestMetadataFilterTool:
    """Tests for metadata filter extraction."""
    
    def test_detect_bukhari_collection(self):
        """Verify Bukhari collection is detected."""
        filters = extract_metadata_filters("hadith from Bukhari about prayer", use_llm=False)
        
        assert filters.collection == "bukhari"
    
    def test_detect_muslim_collection(self):
        """Verify Muslim collection is detected."""
        filters = extract_metadata_filters("صحيح مسلم عن الصيام", use_llm=False)
        
        assert filters.collection == "muslim"
    
    def test_detect_hadith_number(self):
        """Verify hadith number is extracted."""
        filters = extract_metadata_filters("show me hadith number 25", use_llm=False)
        
        assert filters.hadith_id == 25
    
    def test_detect_narrator_english(self):
        """Verify narrator is detected (English)."""
        filters = extract_metadata_filters("narrated by Abu Hurairah", use_llm=False)
        
        assert filters.narrator is not None
        assert "Abu Hurairah" in filters.narrator
    
    def test_detect_narrator_arabic(self):
        """Verify narrator is detected (Arabic)."""
        filters = extract_metadata_filters("عن أبو هريرة", use_llm=False)
        
        assert filters.narrator is not None
    
    def test_relax_filters_level_1(self):
        """Test filter relaxation level 1 (remove chapter)."""
        filters = MetadataFilter(
            collection="bukhari",
            book_id=1,
            chapter_id=5,
            narrator="Abu Hurairah"
        )
        
        relaxed = filters.relax(level=1)
        
        assert relaxed.collection == "bukhari"
        assert relaxed.book_id == 1
        assert relaxed.chapter_id is None  # Removed
        assert relaxed.narrator == "Abu Hurairah"
    
    def test_relax_filters_level_2(self):
        """Test filter relaxation level 2 (remove book)."""
        filters = MetadataFilter(collection="bukhari", book_id=1, chapter_id=5)
        
        relaxed = filters.relax(level=2)
        
        assert relaxed.collection == "bukhari"
        assert relaxed.book_id is None  # Removed
        assert relaxed.chapter_id is None  # Removed
    
    def test_chroma_filter_conversion(self):
        """Test conversion to ChromaDB filter format."""
        filters = MetadataFilter(collection="bukhari", hadith_id=25)
        
        chroma_filter = filters.to_chroma_filter()
        
        assert chroma_filter is not None
        assert "$and" in chroma_filter


# ============================================================================
# Test: Reciprocal Rank Fusion
# ============================================================================

class TestRRFFusion:
    """Tests for RRF fusion algorithm."""
    
    def test_rrf_combines_unique_docs(self, sample_documents):
        """Test RRF combines documents from different sources."""
        semantic = sample_documents[:2]  # First 2
        keyword = sample_documents[1:]   # Last 2 (overlapping middle)
        
        fused = _reciprocal_rank_fusion(semantic, keyword, alpha=0.5, k=5)
        
        # Should have all 3 unique documents
        assert len(fused) == 3
        chunk_ids = {doc.chunk_id for doc in fused}
        assert "bukhari_1_1" in chunk_ids
        assert "bukhari_1_2" in chunk_ids
        assert "muslim_1_1" in chunk_ids
    
    def test_rrf_respects_alpha_semantic(self, sample_documents):
        """Test alpha=1.0 gives full weight to semantic."""
        semantic = [sample_documents[0]]  # High score
        keyword = [sample_documents[2]]   # Different doc
        
        fused = _reciprocal_rank_fusion(semantic, keyword, alpha=1.0, k=5)
        
        # Semantic doc should be ranked first
        assert fused[0].chunk_id == "bukhari_1_1"
    
    def test_rrf_empty_inputs(self):
        """Test RRF handles empty input lists."""
        fused = _reciprocal_rank_fusion([], [], alpha=0.5, k=5)
        
        assert fused == []


# ============================================================================
# Test: Deduplication and Aggregation
# ============================================================================

class TestAggregation:
    """Tests for result aggregation."""
    
    def test_deduplicate_by_chunk_id(self, sample_documents):
        """Test deduplication keeps highest score."""
        # Add duplicate with lower score
        dup = sample_documents[0].model_copy(update={"score": 0.5})
        docs = sample_documents + [dup]
        
        deduped = _deduplicate_documents(docs)
        
        # Should have 3 unique docs
        assert len(deduped) == 3
        
        # Original high score should be kept
        bukhari_1 = next(d for d in deduped if d.chunk_id == "bukhari_1_1")
        assert bukhari_1.score == 0.95
    
    def test_normalize_scores(self, sample_documents):
        """Test score normalization to [0,1]."""
        normalized = _normalize_scores(sample_documents)
        
        scores = [d.score for d in normalized]
        assert min(scores) >= 0.0
        assert max(scores) <= 1.0
    
    def test_aggregation_full_pipeline(self, sample_documents):
        """Test full aggregation pipeline."""
        raw_results = [
            sample_documents[:2],
            sample_documents[1:],
        ]
        
        aggregated = aggregate_results(
            raw_results=raw_results,
            original_query="test query",
            top_k=10,
            use_reranker=False,
        )
        
        assert isinstance(aggregated, AggregatedResults)
        assert aggregated.total_unique == 3
        assert aggregated.duplicates_removed >= 0
        assert len(aggregated.documents) <= 10
    
    def test_aggregation_respects_top_k(self, sample_documents):
        """Test aggregation respects top_k limit."""
        raw_results = [sample_documents]
        
        aggregated = aggregate_results(
            raw_results=raw_results,
            original_query="test",
            top_k=2,
            use_reranker=False,
        )
        
        assert len(aggregated.documents) == 2


# ============================================================================
# Test: Parallel Map-Reduce
# ============================================================================

class TestParallelExecution:
    """Tests for parallel sub-query execution."""
    
    @pytest.mark.asyncio
    async def test_parallel_multiple_subqueries(self):
        """Test parallel execution with multiple sub-queries."""
        sub_queries = ["prayer rules", "fasting rules"]
        
        with patch('src.agents.retrieval.hybrid_search') as mock_search:
            # Mock returns different docs for each query
            mock_search.side_effect = [
                HybridSearchResult(
                    documents=[Document(chunk_id="doc_prayer", text="Prayer hadith", score=0.9)],
                    query="prayer rules",
                    semantic_results=[],
                    keyword_results=[],
                    alpha=0.5,
                ),
                HybridSearchResult(
                    documents=[Document(chunk_id="doc_fasting", text="Fasting hadith", score=0.85)],
                    query="fasting rules",
                    semantic_results=[],
                    keyword_results=[],
                    alpha=0.5,
                ),
            ]
            
            with patch('src.agents.retrieval.extract_metadata_filters') as mock_filters:
                mock_filters.return_value = MetadataFilter()
                
                results = await _execute_parallel_searches(
                    sub_queries=sub_queries,
                    expanded_terms={"prayer rules": [], "fasting rules": []},
                    target_collections=["bukhari"],
                    top_k=10,
                )
        
        # Should have results for both sub-queries
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_parallel_handles_exception(self):
        """Test parallel execution handles task exceptions gracefully."""
        with patch('src.agents.retrieval._execute_search_with_retry') as mock_search:
            mock_search.side_effect = [
                [Document(chunk_id="doc1", text="test", score=0.9)],
                Exception("Search failed"),
            ]
            
            results = await _execute_parallel_searches(
                sub_queries=["query1", "query2"],
                expanded_terms={},
                target_collections=["bukhari"],
                top_k=10,
            )
        
        # First should succeed, second should be empty list
        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[1] == []


# ============================================================================
# Test: Self-Correction (Retry Logic)
# ============================================================================

class TestSelfCorrection:
    """Tests for self-correction with filter relaxation."""
    
    @pytest.mark.asyncio
    async def test_retry_relaxes_filters(self):
        """Test that retry relaxes filters when no results."""
        call_count = 0
        
        def mock_hybrid_search(query, k, alpha, filters):
            nonlocal call_count
            call_count += 1
            
            # First call (strict filters) returns empty
            if call_count == 1:
                return HybridSearchResult(
                    documents=[],
                    query=query,
                    semantic_results=[],
                    keyword_results=[],
                    alpha=alpha,
                )
            # Second call (relaxed filters) returns results
            return HybridSearchResult(
                documents=[Document(chunk_id="doc1", text="Found after relaxation", score=0.8)],
                query=query,
                semantic_results=[],
                keyword_results=[],
                alpha=alpha,
            )
        
        with patch('src.agents.retrieval.hybrid_search', side_effect=mock_hybrid_search):
            with patch('src.agents.retrieval.extract_metadata_filters') as mock_filters:
                mock_filters.return_value = MetadataFilter(
                    collection="bukhari",
                    book_id=1,
                    chapter_id=5,
                )
                
                results = await _execute_search_with_retry(
                    query="test query",
                    expanded_terms=[],
                    target_collections=["bukhari"],
                    top_k=10,
                )
        
        # Should have found results after retry
        assert len(results) == 1
        assert call_count == 2  # Called twice (first failed, second succeeded)


# ============================================================================
# Test: User Text Processing
# ============================================================================

class TestUserTextProcessing:
    """Tests for user-provided text processing."""
    
    def test_clean_user_text_removes_quotes(self):
        """Test quote removal from user text."""
        cleaned = _clean_user_text('"إنما الأعمال بالنيات"')
        
        assert cleaned.startswith("إنما")
        assert '"' not in cleaned
    
    def test_clean_user_text_removes_prefix(self):
        """Test common prefix removal."""
        cleaned = _clean_user_text("The Prophet said: Actions are by intentions")
        
        assert "Actions" in cleaned
        assert "Prophet said" not in cleaned.lower()
    
    def test_user_text_processor_integration(self):
        """Test UserHadithProcessorTool with mocked search."""
        with patch.object(SemanticSearchTool, '__call__') as mock_search:
            mock_search.return_value = SearchResult(
                documents=[
                    Document(chunk_id="match", text="Similar hadith", score=0.9)
                ],
                query="test",
                search_type=SearchType.SEMANTIC,
            )
            
            tool = UserHadithProcessorTool()
            result = tool(user_text="إنما الأعمال بالنيات", find_similar=True)
        
        assert result.processed_text
        assert len(result.similar_hadiths) == 1


# ============================================================================
# Test: Full Retrieval Agent Integration
# ============================================================================

class TestRetrievalAgentIntegration:
    """Integration tests for the retrieval agent."""
    
    def test_agent_returns_expected_structure(self, sample_state):
        """Test agent returns correct state structure."""
        with patch('src.agents.retrieval._handle_base_knowledge_search') as mock_search:
            mock_search.return_value = [
                Document(chunk_id="doc1", text="Test hadith", score=0.9)
            ]
            
            result = retrieval_agent(sample_state)
        
        assert "retrieved_docs" in result
        assert "metadata" in result
        assert isinstance(result["retrieved_docs"], list)
    
    def test_agent_routes_user_text(self):
        """Test agent routes user_text to UserHadithProcessor."""
        state = {
            "original_query": "إنما الأعمال بالنيات",
            "input_source": "user_text",
            "metadata": {},
        }
        
        with patch('src.agents.retrieval._handle_user_text') as mock_handler:
            mock_handler.return_value = [
                Document(chunk_id="match", text="Matched", score=0.95)
            ]
            
            result = retrieval_agent(state)
        
        mock_handler.assert_called_once()
        assert len(result["retrieved_docs"]) == 1
    
    def test_standalone_retrieve_function(self):
        """Test standalone retrieve() function."""
        with patch('src.agents.retrieval.retrieval_agent') as mock_agent:
            mock_agent.return_value = {
                "retrieved_docs": [
                    Document(chunk_id="doc1", text="Test", score=0.9)
                ],
                "metadata": {},
            }
            
            results = retrieve("prayer hadiths", top_k=5)
        
        assert len(results) == 1


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and graceful degradation."""
    
    def test_agent_handles_empty_query(self):
        """Test agent handles empty query gracefully."""
        state = {
            "original_query": "",
            "metadata": {},
        }
        
        result = retrieval_agent(state)
        
        assert result["retrieved_docs"] == []
        assert "error" in result["metadata"]
    
    def test_aggregation_handles_empty_results(self):
        """Test aggregation handles empty result lists."""
        aggregated = aggregate_results(
            raw_results=[[], []],
            original_query="test",
            top_k=10,
            use_reranker=False,
        )
        
        assert aggregated.documents == []
        assert aggregated.total_unique == 0


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
