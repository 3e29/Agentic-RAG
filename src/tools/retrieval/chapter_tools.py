"""
Chapter Lookup Tools for Hadith RAG System

This module provides tools for mapping user subject terms to chapter IDs
for precise filtering in hadith retrieval.

**Key Features:**
- Loads chapter data from raw JSON files (Bukhari, Muslim)
- Singleton pattern for efficient memory usage
- Query expansion for English synonym generation
- Substring matching against chapter titles
- Fallback to no filter if no match found

**Usage:**
```python
from src.tools.retrieval.chapter_tools import find_chapter_for_subject

result = find_chapter_for_subject("البيع", collection="bukhari")
# Returns: {"chapter_id": 34, "chapter_title_en": "Sales and Trade", 
#           "chapter_title_ar": "كتاب البيوع", "confidence": 0.95}
```
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langsmith import traceable

from src.tools.retrieval.filter_tools import QueryExpansionTool

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class ChapterInfo(BaseModel):
    """Information about a single chapter."""
    id: int = Field(..., description="Chapter ID")
    book_id: int = Field(..., description="Book/Collection ID (1=Bukhari, 2=Muslim)")
    arabic: str = Field(..., description="Arabic chapter title")
    english: str = Field(..., description="English chapter title")
    collection: str = Field(..., description="Collection name (bukhari/muslim)")


class ChapterLookupResult(BaseModel):
    """Result from chapter lookup."""
    chapter_id: int = Field(..., description="Found chapter ID")
    chapter_title_en: str = Field(..., description="English chapter title")
    chapter_title_ar: str = Field(..., description="Arabic chapter title")
    collection: str = Field(..., description="Collection name")
    confidence: float = Field(..., description="Match confidence (0.0-1.0)")
    match_type: str = Field(..., description="How the match was found")


# ============================================================================
# Chapter Index Singleton
# ============================================================================

class ChapterIndex:
    """
    Singleton class that loads and indexes chapter data from raw JSON files.
    
    Provides fast lookup for:
    - Chapter by ID
    - Chapter by title (Arabic/English)
    - Subject term to chapter mapping
    """
    
    _instance: Optional["ChapterIndex"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "ChapterIndex":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ChapterIndex._initialized:
            return
        
        self.chapters: Dict[str, Dict[int, ChapterInfo]] = {
            "bukhari": {},
            "muslim": {},
        }
        self.title_index: Dict[str, List[Tuple[int, str, str]]] = {
            "bukhari": [],  # List of (chapter_id, english_title_lower, arabic_title)
            "muslim": [],
        }
        
        self._load_chapters()
        ChapterIndex._initialized = True
    
    def _load_chapters(self):
        """Load chapter data from raw JSON files."""
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw"
        
        # Load Bukhari
        bukhari_path = data_dir / "bukhari.json"
        if bukhari_path.exists():
            self._load_collection(bukhari_path, "bukhari")
        else:
            logger.warning(f"Bukhari data file not found: {bukhari_path}")
        
        # Load Muslim
        muslim_path = data_dir / "muslim.json"
        if muslim_path.exists():
            self._load_collection(muslim_path, "muslim")
        else:
            logger.warning(f"Muslim data file not found: {muslim_path}")
        
        logger.info(
            f"Chapter index loaded: Bukhari={len(self.chapters['bukhari'])}, "
            f"Muslim={len(self.chapters['muslim'])}"
        )
    
    def _load_collection(self, file_path: Path, collection: str):
        """Load chapters from a single collection file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            chapters_data = data.get("chapters", [])
            
            for chapter in chapters_data:
                chapter_id = chapter.get("id")
                if chapter_id is None:
                    continue
                
                info = ChapterInfo(
                    id=chapter_id,
                    book_id=chapter.get("bookId", 1 if collection == "bukhari" else 2),
                    arabic=chapter.get("arabic", ""),
                    english=chapter.get("english", ""),
                    collection=collection,
                )
                
                self.chapters[collection][chapter_id] = info
                
                # Add to title index for fast lookup
                self.title_index[collection].append((
                    chapter_id,
                    info.english.lower(),
                    info.arabic,
                ))
            
            logger.debug(f"Loaded {len(self.chapters[collection])} chapters from {collection}")
            
        except Exception as e:
            logger.error(f"Failed to load chapters from {file_path}: {e}")
    
    def get_chapter(self, chapter_id: int, collection: str = "bukhari") -> Optional[ChapterInfo]:
        """Get chapter info by ID."""
        return self.chapters.get(collection, {}).get(chapter_id)
    
    def search_by_title(
        self,
        search_terms: List[str],
        collection: Optional[str] = None,
    ) -> List[Tuple[ChapterInfo, float, str]]:
        """
        Search for chapters by title using substring matching.
        
        Args:
            search_terms: List of terms to search for (English, lowercase)
            collection: Optional collection to limit search
            
        Returns:
            List of (ChapterInfo, confidence, match_type) tuples
        """
        results = []
        collections_to_search = [collection] if collection else ["bukhari", "muslim"]
        
        for coll in collections_to_search:
            if coll not in self.title_index:
                continue
            
            for chapter_id, english_lower, arabic in self.title_index[coll]:
                for term in search_terms:
                    term_lower = term.lower().strip() if isinstance(term, str) else str(term).lower().strip()
                    if not term_lower:
                        continue
                    
                    # Check English title
                    if term_lower in english_lower:
                        # Calculate confidence based on match quality
                        if english_lower == term_lower:
                            confidence = 1.0
                            match_type = "exact_english"
                        elif english_lower.startswith(term_lower) or english_lower.endswith(term_lower):
                            confidence = 0.9
                            match_type = "prefix_suffix_english"
                        else:
                            # Substring match - confidence based on term length ratio
                            confidence = min(0.85, len(term_lower) / len(english_lower) + 0.3)
                            match_type = "substring_english"
                        
                        chapter_info = self.chapters[coll][chapter_id]
                        results.append((chapter_info, confidence, match_type))
                        break  # Found match for this chapter, move to next
                    
                    # Check Arabic title
                    if term in arabic:
                        if arabic == term:
                            confidence = 1.0
                            match_type = "exact_arabic"
                        elif arabic.startswith(term) or arabic.endswith(term):
                            confidence = 0.9
                            match_type = "prefix_suffix_arabic"
                        else:
                            confidence = min(0.85, len(term) / len(arabic) + 0.3)
                            match_type = "substring_arabic"
                        
                        chapter_info = self.chapters[coll][chapter_id]
                        results.append((chapter_info, confidence, match_type))
                        break
        
        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# Global singleton instance
_chapter_index: Optional[ChapterIndex] = None


def get_chapter_index() -> ChapterIndex:
    """Get or create the chapter index singleton."""
    global _chapter_index
    if _chapter_index is None:
        _chapter_index = ChapterIndex()
    return _chapter_index


# ============================================================================
# Chapter Lookup Tool
# ============================================================================

class ChapterLookupTool:
    """
    Tool for finding chapter IDs from subject terms.
    
    Uses query expansion to generate English synonyms, then performs
    substring matching against chapter titles.
    
    **Process:**
    1. Extract potential subject terms from query
    2. Expand to English synonyms using QueryExpansionTool
    3. Search chapter titles for matches
    4. Return best match with confidence score
    """
    
    name: str = "chapter_lookup"
    description: str = "Find chapter IDs from subject terms or chapter names"
    
    def __init__(self):
        self.index = get_chapter_index()
        self.expander = QueryExpansionTool()
    
    @traceable(name="chapter_lookup_tool")
    def __call__(
        self,
        subject_term: str,
        collection: Optional[str] = None,
    ) -> Optional[ChapterLookupResult]:
        """
        Find the chapter ID for a given subject term.
        
        Args:
            subject_term: The subject/topic to find (e.g., "البيع", "sales", "Zakat")
            collection: Optional collection to limit search ("bukhari", "muslim")
            
        Returns:
            ChapterLookupResult if found, None otherwise
        """
        if not subject_term or not subject_term.strip():
            logger.warning("Empty subject term provided")
            return None
        
        subject_term = subject_term.strip()
        logger.info(f"Looking up chapter for subject: '{subject_term}'")
        
        # Step 1: Generate search terms
        search_terms = self._generate_search_terms(subject_term)
        logger.debug(f"Generated search terms: {search_terms}")
        
        if not search_terms:
            logger.warning(f"No search terms generated for: {subject_term}")
            return None
        
        # Step 2: Search chapter index
        matches = self.index.search_by_title(search_terms, collection)
        
        if not matches:
            logger.info(f"No chapter match found for: {subject_term}")
            return None
        
        # Step 3: Return best match
        best_match, confidence, match_type = matches[0]
        
        result = ChapterLookupResult(
            chapter_id=best_match.id,
            chapter_title_en=best_match.english,
            chapter_title_ar=best_match.arabic,
            collection=best_match.collection,
            confidence=confidence,
            match_type=match_type,
        )
        
        logger.info(
            f"Found chapter: {result.chapter_id} ({result.chapter_title_en}) "
            f"with confidence {confidence:.2f} via {match_type}"
        )
        
        return result
    
    def _generate_search_terms(self, subject_term: str) -> List[str]:
        """
        Generate search terms from the subject.
        
        Uses query expansion to get English synonyms for better matching.
        """
        search_terms = [subject_term]
        
        # Add the original term without "كتاب" prefix if present
        if subject_term.startswith("كتاب "):
            search_terms.append(subject_term[5:])
        
        # Add common variations
        # Remove "ال" prefix for Arabic
        if subject_term.startswith("ال"):
            search_terms.append(subject_term[2:])
        
        # Try query expansion to get English synonyms
        try:
            expansion = self.expander(subject_term, use_llm=False)
            
            # Add expanded terms
            if expansion.expanded_terms:
                search_terms.extend(expansion.expanded_terms)
            
            # Add translations
            if expansion.translations:
                for orig, trans in expansion.translations.items():
                    if isinstance(trans, list):
                        search_terms.extend(trans)
                    else:
                        search_terms.append(trans)
                        
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            term_key = term.lower().strip() if isinstance(term, str) else str(term).lower().strip()
            if term_key and term_key not in seen:
                seen.add(term_key)
                unique_terms.append(term)
        
        return unique_terms


# ============================================================================
# Convenience Functions
# ============================================================================

@traceable(name="find_chapter_for_subject")
def find_chapter_for_subject(
    subject_term: str,
    collection: Optional[str] = None,
) -> Optional[ChapterLookupResult]:
    """
    Convenience function to find chapter ID for a subject term.
    
    Args:
        subject_term: The subject/topic (e.g., "البيع", "sales", "Zakat")
        collection: Optional collection filter ("bukhari", "muslim")
        
    Returns:
        ChapterLookupResult with chapter_id, titles, and confidence
        None if no match found
    """
    tool = ChapterLookupTool()
    return tool(subject_term, collection)


def get_all_chapters(collection: Optional[str] = None) -> List[ChapterInfo]:
    """
    Get all chapters, optionally filtered by collection.
    
    Useful for debugging and testing.
    """
    index = get_chapter_index()
    
    if collection:
        return list(index.chapters.get(collection, {}).values())
    
    all_chapters = []
    for coll_chapters in index.chapters.values():
        all_chapters.extend(coll_chapters.values())
    return all_chapters


# ============================================================================
# Testing Entry Point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test cases
    test_subjects = [
        "البيع",
        "sales",
        "Zakat",
        "الزكاة",
        "prayer",
        "الصلاة",
        "fasting",
        "الصيام",
        "faith",
        "الإيمان",
    ]
    
    print("=" * 60)
    print("Chapter Lookup Tool Test")
    print("=" * 60)
    
    for subject in test_subjects:
        result = find_chapter_for_subject(subject, collection="bukhari")
        if result:
            print(f"\n'{subject}' -> Chapter {result.chapter_id}: {result.chapter_title_en}")
            print(f"  Arabic: {result.chapter_title_ar}")
            print(f"  Confidence: {result.confidence:.2f} ({result.match_type})")
        else:
            print(f"\n'{subject}' -> No match found")
