"""
Comprehensive Test Suite for Query Analysis Agent

This module provides strict pytest-based tests for the Query Analysis Agent
and all its tools following Clean Architecture principles.

Test Categories:
1. Unit Tests - Individual tool testing
2. Integration Tests - Full pipeline testing
3. Optimization Tests - Conditional execution verification

Run with: pytest tests/test_query_analysis.py -v
"""

import pytest
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.query_processing import (
    query_normalization_tool,
    input_source_identification_tool,
    collection_target_detection_tool,
    typo_correction_tool,
    intent_classification_tool,
    query_decomposition_tool,
    QueryNormalizationOutput,
    InputSourceOutput,
    CollectionTargetOutput,
    TypoCorrectionOutput,
    IntentClassificationOutput,
    QueryDecompositionOutput,
)
from src.agents.query_analysis import analyze_query, query_analysis_agent

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Unit Tests: Query Normalization Tool (No LLM)
# ============================================================================

class TestQueryNormalizationTool:
    """Test suite for the Query Normalization Tool (regex-based, no LLM)."""
    
    def test_normalize_alif_variants(self):
        """Test normalization of Alif variants (أ, إ, آ, ٱ) -> ا"""
        query = "أحاديث إسلامية آمنة"
        result = query_normalization_tool(query)
        
        assert isinstance(result, QueryNormalizationOutput)
        assert "أ" not in result.normalized_text
        assert "إ" not in result.normalized_text
        assert "آ" not in result.normalized_text
        assert "ا" in result.normalized_text
        assert "Normalized Alif variants" in str(result.transformations_applied)
    
    def test_remove_tashkeel(self):
        """Test removal of Tashkeel (diacritical marks)."""
        query = "أَحَادِيثُ الصَّلَاةِ"
        result = query_normalization_tool(query)
        
        assert isinstance(result, QueryNormalizationOutput)
        # Check that diacritics are removed
        assert "َ" not in result.normalized_text  # Fatha
        assert "ُ" not in result.normalized_text  # Damma
        assert "ِ" not in result.normalized_text  # Kasra
        assert "Removed Tashkeel" in str(result.transformations_applied)
    
    def test_remove_tatweel(self):
        """Test removal of Tatweel (kashida elongation)."""
        query = "أحاديث عن الـصــلاة"
        result = query_normalization_tool(query)
        
        assert isinstance(result, QueryNormalizationOutput)
        assert "ـ" not in result.normalized_text
        assert "Removed Tatweel" in str(result.transformations_applied)
    
    def test_normalize_teh_marbuta(self):
        """Test normalization of Teh Marbuta (ة -> ه)."""
        query = "أحاديث الصلاة"
        result = query_normalization_tool(query)
        
        assert isinstance(result, QueryNormalizationOutput)
        assert "ة" not in result.normalized_text
        assert "ه" in result.normalized_text
        assert "Normalized Teh Marbuta" in str(result.transformations_applied)
    
    def test_full_arabic_normalization(self):
        """Test complete Arabic text normalization."""
        query = "أحاديث عن الـصــلاة"
        result = query_normalization_tool(query)
        
        # Expected: "احاديث عن الصلاه"
        assert isinstance(result, QueryNormalizationOutput)
        expected = "احاديث عن الصلاه"
        assert result.normalized_text == expected
    
    def test_english_query_no_change(self):
        """Test that English text passes through with minimal changes."""
        query = "What are hadiths about prayer?"
        result = query_normalization_tool(query)
        
        assert isinstance(result, QueryNormalizationOutput)
        assert result.normalized_text == query.strip()
        assert "No normalization needed" in str(result.transformations_applied)
    
    def test_whitespace_normalization(self):
        """Test normalization of multiple whitespaces."""
        query = "hadiths   about    prayer"
        result = query_normalization_tool(query)
        
        assert isinstance(result, QueryNormalizationOutput)
        assert "   " not in result.normalized_text
        assert result.normalized_text == "hadiths about prayer"


# ============================================================================
# Unit Tests: Collection Target Detection Tool (No LLM)
# ============================================================================

class TestCollectionTargetDetectionTool:
    """Test suite for the Collection Target Detection Tool (keyword-based)."""
    
    def test_detect_bukhari_english(self):
        """Test detection of Bukhari collection (English)."""
        query = "What does Bukhari say about prayer?"
        result = collection_target_detection_tool(query)
        
        assert isinstance(result, CollectionTargetOutput)
        assert "bukhari" in result.targets
        assert len(result.targets) == 1
    
    def test_detect_bukhari_arabic(self):
        """Test detection of Bukhari collection (Arabic)."""
        query = "ما يقول البخاري عن الصلاة"
        result = collection_target_detection_tool(query)
        
        assert isinstance(result, CollectionTargetOutput)
        assert "bukhari" in result.targets
    
    def test_detect_muslim_english(self):
        """Test detection of Muslim collection (English)."""
        query = "What did Muslim say about fasting?"
        result = collection_target_detection_tool(query)
        
        assert isinstance(result, CollectionTargetOutput)
        assert "muslim" in result.targets
    
    def test_detect_muslim_arabic(self):
        """Test detection of Muslim collection (Arabic)."""
        query = "أحاديث صحيح مسلم عن الصيام"
        result = collection_target_detection_tool(query)
        
        assert isinstance(result, CollectionTargetOutput)
        assert "muslim" in result.targets
    
    def test_detect_both_collections(self):
        """Test detection when both collections are mentioned."""
        query = "Compare Bukhari and Muslim on charity"
        result = collection_target_detection_tool(query)
        
        assert isinstance(result, CollectionTargetOutput)
        assert "bukhari" in result.targets
        assert "muslim" in result.targets
        assert len(result.targets) == 2
    
    def test_no_specific_collection_returns_all(self):
        """Test that no specific mention returns all collections."""
        query = "What are hadiths about prayer?"
        result = collection_target_detection_tool(query)
        
        assert isinstance(result, CollectionTargetOutput)
        assert "bukhari" in result.targets
        assert "muslim" in result.targets
        assert "No specific collection mentioned" in result.reasoning


# ============================================================================
# Unit Tests: Input Source Identification Tool (LLM-based)
# ============================================================================

class TestInputSourceIdentificationTool:
    """Test suite for the Input Source Identification Tool."""
    
    def test_base_knowledge_query(self):
        """Test identification of database query (base_knowledge)."""
        query = "What are hadiths about prayer?"
        result = input_source_identification_tool(query)
        
        assert isinstance(result, InputSourceOutput)
        assert result.source_type == "base_knowledge"
        assert result.confidence > 0.0
    
    def test_user_text_with_explicit_pattern(self):
        """Test identification of user-provided text."""
        query = "Explain this text: 'Actions are judged by intentions'"
        result = input_source_identification_tool(query)
        
        assert isinstance(result, InputSourceOutput)
        assert result.source_type == "user_text"
        assert result.confidence >= 0.5
    
    def test_user_text_with_analyze_pattern(self):
        """Test identification with 'analyze' keyword."""
        query = "Analyze the following hadith passage for me"
        result = input_source_identification_tool(query)
        
        assert isinstance(result, InputSourceOutput)
        assert result.source_type == "user_text"


# ============================================================================
# Unit Tests: Intent Classification Tool (LLM-based)
# ============================================================================

class TestIntentClassificationTool:
    """Test suite for the Intent Classification Tool."""
    
    def test_thematic_search_intent(self):
        """Test classification of thematic search query."""
        query = "What are hadiths about prayer?"
        result = intent_classification_tool(query)
        
        assert isinstance(result, IntentClassificationOutput)
        assert result.intent == "thematic_search"
        assert result.confidence > 0.0
    
    def test_specific_lookup_intent_with_number(self):
        """Test classification of specific lookup by hadith number."""
        query = "Hadith #1 in Bukhari"
        result = intent_classification_tool(query)
        
        assert isinstance(result, IntentClassificationOutput)
        assert result.intent == "specific_lookup"
    
    def test_specific_lookup_intent_with_text(self):
        """Test classification of specific lookup by hadith text."""
        query = "Find the hadith: إنما الأعمال بالنيات"
        result = intent_classification_tool(query)
        
        assert isinstance(result, IntentClassificationOutput)
        assert result.intent == "specific_lookup"
    
    def test_comparative_analysis_intent(self):
        """Test classification of comparative analysis query."""
        query = "Compare what Bukhari and Muslim say about charity"
        result = intent_classification_tool(query)
        
        assert isinstance(result, IntentClassificationOutput)
        assert result.intent == "comparative_analysis"


# ============================================================================
# Integration Tests: Full Pipeline
# ============================================================================

class TestFullPipeline:
    """Integration tests for the complete Query Analysis pipeline."""
    
    def test_simple_english_query(self):
        """Test full pipeline with simple English query."""
        result = analyze_query("What are hadiths about prayer?")
        
        assert result["original_query"] == "What are hadiths about prayer?"
        assert result["normalized_query"] is not None
        assert result["corrected_query"] is not None
        assert result["input_source"] == "base_knowledge"
        assert result["query_intent"] == "thematic_search"
        assert result["target_collections"] is not None
        assert result["language"] in ["ar", "en", "mixed"]
    
    def test_arabic_query_normalization(self):
        """Test that Arabic query is properly normalized through pipeline."""
        result = analyze_query("أحاديث عن الـصــلاة")
        
        assert result["original_query"] == "أحاديث عن الـصــلاة"
        # Normalized should have tatweel removed
        assert "ـ" not in result["normalized_query"]
    
    def test_complex_arabic_query(self):
        """Test full pipeline with complex Arabic query."""
        result = analyze_query("ما هي احاديث الصلاة والزكاة وما الفرق بينهما")
        
        assert result["query_intent"] == "comparative_analysis"
        # Should be decomposed for comparative analysis
        if result["sub_queries"] is not None:
            assert len(result["sub_queries"]) > 0
    
    def test_specific_hadith_lookup(self):
        """Test pipeline with specific hadith lookup."""
        result = analyze_query("Show me hadith number 1 from Sahih Bukhari")
        
        assert result["query_intent"] == "specific_lookup"
        assert "bukhari" in result["target_collections"]
    
    def test_metadata_completeness(self):
        """Test that metadata contains all expected fields."""
        result = analyze_query("What are hadiths about prayer?")
        
        qa_meta = result["metadata"]["query_analysis"]
        assert "stages_completed" in qa_meta
        assert "stages_skipped" in qa_meta
        assert "errors" in qa_meta
        assert "pipeline_version" in qa_meta


# ============================================================================
# Optimization Tests: Conditional Execution Verification
# ============================================================================

class TestConditionalExecution:
    """Tests to verify the conditional execution optimization."""
    
    def test_specific_lookup_skips_decomposition(self):
        """
        OPTIMIZATION TEST: Verify that specific_lookup queries skip decomposition.
        
        When intent == specific_lookup, the decomposition tool should be skipped
        to save tokens and reduce latency.
        """
        result = analyze_query("Hadith #1 in Bukhari")
        
        # Verify intent is specific_lookup
        assert result["query_intent"] == "specific_lookup"
        
        # Verify decomposition was skipped
        assert result["sub_queries"] is None
        
        # Verify metadata shows skipped
        decomp_meta = result["metadata"]["query_analysis"]["query_decomposition"]
        assert decomp_meta["skipped"] == True
        assert "specific_lookup" in decomp_meta["skip_reason"]
        
        # Verify it's in skipped stages
        assert "query_decomposition" in result["metadata"]["query_analysis"]["stages_skipped"]
    
    def test_user_text_skips_decomposition(self):
        """
        OPTIMIZATION TEST: Verify that user_text input skips decomposition.
        
        When input_source != base_knowledge, decomposition should be skipped.
        """
        result = analyze_query("Explain this text: 'Actions are judged by intentions'")
        
        # Verify input source is user_text
        assert result["input_source"] == "user_text"
        
        # Verify decomposition was skipped
        assert result["sub_queries"] is None
        
        # Verify metadata shows skipped
        decomp_meta = result["metadata"]["query_analysis"]["query_decomposition"]
        assert decomp_meta["skipped"] == True
    
    def test_thematic_search_runs_decomposition(self):
        """
        Verify that thematic_search with base_knowledge runs decomposition.
        """
        result = analyze_query("What are hadiths about prayer and fasting?")
        
        # Should be thematic_search or comparative_analysis
        assert result["query_intent"] in ["thematic_search", "comparative_analysis"]
        assert result["input_source"] == "base_knowledge"
        
        # Decomposition should have run (not skipped)
        decomp_meta = result["metadata"]["query_analysis"]["query_decomposition"]
        assert decomp_meta["skipped"] == False
    
    def test_comparative_analysis_runs_decomposition(self):
        """
        Verify that comparative_analysis with base_knowledge runs decomposition.
        """
        result = analyze_query("Compare Bukhari and Muslim on prayer")
        
        assert result["query_intent"] == "comparative_analysis"
        assert result["input_source"] == "base_knowledge"
        
        # Decomposition should have run
        decomp_meta = result["metadata"]["query_analysis"]["query_decomposition"]
        assert decomp_meta["skipped"] == False


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests to verify defensive programming and fallback values."""
    
    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError):
            analyze_query("")
    
    def test_whitespace_only_query_raises_error(self):
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError):
            analyze_query("   ")
    
    def test_pipeline_never_crashes(self):
        """Test that pipeline handles various edge cases without crashing."""
        edge_cases = [
            "x",  # Very short
            "a" * 1000,  # Very long
            "🙏 prayer",  # Emoji
            "123 456",  # Numbers only
            "مع 123 and text",  # Mixed everything
        ]
        
        for query in edge_cases:
            try:
                result = analyze_query(query)
                assert result is not None
                assert result["corrected_query"] is not None
            except ValueError:
                # Empty/invalid queries may raise ValueError, which is acceptable
                pass


# ============================================================================
# Regression Tests (Based on Original Test Cases)
# ============================================================================

class TestRegressionCases:
    """Regression tests based on original test_query_analysis.py cases."""
    
    def test_english_query_with_typo(self):
        """Test handling of English query with typos."""
        result = analyze_query("What dose Islam say abou honesty?")
        
        assert result["query_intent"] == "thematic_search"
        # Typo correction should have fixed the query
        corrections = result["metadata"]["query_analysis"]["typo_correction"]["corrections_made"]
        assert len(corrections) > 0 or result["corrected_query"] != result["original_query"]
    
    def test_specific_arabic_hadith_lookup(self):
        """Test specific Arabic hadith lookup."""
        result = analyze_query("احاديث ابو هريرة عن القطط")
        
        assert result["query_intent"] == "specific_lookup"
        assert result["language"] in ["ar", "mixed"]


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
