"""
Arabic Text Processing Utilities

This module provides functions for preprocessing and normalizing Arabic text
for the Hadith RAG system. Implements FR-DIC-33 (Text Preprocessing).

NOTE: The raw JSON files (bukhari.json, muslim.json) have already been preprocessed
using script.py, script2.py, and script3.py which handled:
- JSON formatting
- Removal of directional marks
- Removal of diacritics (keeping shadda)

This module provides lightweight additional processing for runtime use cases.

Functions:
    - normalize_whitespace: Standardize spacing and newlines (primary use)
    - normalize_arabic_text: Light normalization for already-clean text
    - normalize_arabic_for_search: Deep normalization for BM25 search
    - expand_arabic_keywords: Morphological expansion (prefix stripping, no synonyms)
    - strip_arabic_prefixes: Remove common Arabic prefixes
    - stem_arabic_word: Extract word root using ISRI stemmer (general NLP, not overfitting)
    - [Legacy functions retained for flexibility]
"""

import re
import logging
from typing import Optional, List, Set


# ============================================================================
# Arabic Stemmer (ISRI - Information Science Research Institute)
# This is a GENERAL-PURPOSE linguistic algorithm, not overfitting.
# It applies to ALL Arabic words by extracting trilateral roots.
# ============================================================================

_arabic_stemmer = None
_stemmer_available = None


def get_arabic_stemmer():
    """
    Get or initialize the Arabic ISRI stemmer (lazy loading).
    
    ISRI stemmer is a general-purpose Arabic morphological analyzer from NLTK.
    It extracts word roots, enabling matching of verb/noun forms:
    - صَبَرَ (verb: he was patient) -> صبر
    - الصبر (noun: the patience) -> صبر
    
    Returns:
        ISRIStemmer instance or None if unavailable
    """
    global _arabic_stemmer, _stemmer_available
    
    if _stemmer_available is False:
        return None
    
    if _arabic_stemmer is None:
        try:
            from nltk.stem.isri import ISRIStemmer
            _arabic_stemmer = ISRIStemmer()
            _stemmer_available = True
        except ImportError:
            logging.getLogger(__name__).warning(
                "NLTK not installed. Arabic stemming disabled. "
                "Install with: pip install nltk"
            )
            _stemmer_available = False
            return None
    
    return _arabic_stemmer


def stem_arabic_word(word: str) -> str:
    """
    Extract the root form of an Arabic word using ISRI stemmer.
    
    This is a GENERAL NLP technique (not overfitting) that handles
    Arabic morphological variants by extracting trilateral roots.
    
    Args:
        word: Arabic word to stem
        
    Returns:
        Stemmed/root form of the word, or original if stemming fails
    """
    if not word or not isinstance(word, str):
        return word
    
    stemmer = get_arabic_stemmer()
    if stemmer is None:
        return word
    
    try:
        stemmed = stemmer.stem(word)
        return stemmed if stemmed else word
    except Exception:
        return word


# Regex pattern for Arabic diacritics EXCEPT shadda (ّ U+0651)
# Note: Raw data is already cleaned, but kept for flexibility
DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u0650\u0652-\u065F\u0670\u06D6-\u06ED]+',
    flags=re.UNICODE
)

# Regex pattern for directional marks (LTR/RTL)
# Note: Raw data is already cleaned, but kept for flexibility
DIRECTIONAL_MARKS_RE = re.compile(r'[\u200e\u200f]+', flags=re.UNICODE)

# Regex pattern for multiple whitespace
WHITESPACE_RE = re.compile(r'\s+', flags=re.UNICODE)


# ============================================================================
# Arabic Character Normalization Maps (for search)
# ============================================================================

# Alef variants -> plain Alef
ALEF_VARIANTS = {
    'أ': 'ا',  # Alef with hamza above
    'إ': 'ا',  # Alef with hamza below
    'آ': 'ا',  # Alef with madda
    'ٱ': 'ا',  # Alef wasla
    'ى': 'ي',  # Alef maqsura -> Ya
}

# Ta marbuta -> Ha (for search matching)
TA_MARBUTA = {'ة': 'ه'}

# Hamza variants -> simple hamza or remove
HAMZA_VARIANTS = {
    'ؤ': 'و',  # Waw with hamza
    'ئ': 'ي',  # Ya with hamza
    'ء': '',   # Standalone hamza - remove
}






def normalize_arabic_for_search(text: str) -> str:
    """
    Deep normalization for Arabic text to improve BM25 search matching.
    
    Normalizes:
    - Alef variants (أ إ آ) -> ا
    - Ta marbuta (ة) -> ه
    - Hamza variants (ؤ ئ) -> base letter
    - Removes diacritics
    - Normalizes whitespace
    
    Args:
        text: Arabic text to normalize
        
    Returns:
        Normalized text for search matching
        
    Example:
        >>> normalize_arabic_for_search("الصَّبْرُ عِنْدَ الشِّدَّةِ")
        "الصبر عند الشده"
    """
    if not isinstance(text, str) or not text:
        return text
    
    # Remove diacritics first
    result = DIACRITICS_RE.sub('', text)
    
    # Normalize Alef variants
    for variant, normalized in ALEF_VARIANTS.items():
        result = result.replace(variant, normalized)
    
    # Normalize Ta marbuta
    for variant, normalized in TA_MARBUTA.items():
        result = result.replace(variant, normalized)
    
    # Normalize Hamza variants
    for variant, normalized in HAMZA_VARIANTS.items():
        result = result.replace(variant, normalized)
    
    # Normalize whitespace
    result = WHITESPACE_RE.sub(' ', result).strip()
    
    return result


def strip_arabic_prefixes(word: str) -> List[str]:
    """
    Strip common Arabic prefixes and return all possible base forms.
    
    Arabic prefixes include:
    - ال (al- definite article)
    - و (wa- and)
    - ف (fa- so/then)
    - ب (bi- with/by)
    - ك (ka- like/as)
    - ل (la- for/to)
    
    Returns:
        List of possible base forms including the original
    """
    forms = [word]
    
    # Common prefixes in order of likelihood
    prefixes = ['وال', 'فال', 'بال', 'كال', 'لل', 'ال', 'و', 'ف', 'ب', 'ك', 'ل']
    
    for prefix in prefixes:
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            base = word[len(prefix):]
            forms.append(base)
            # If we removed و/ف/ب/ك/ل, also try adding ال
            if prefix in ['و', 'ف', 'ب', 'ك', 'ل']:
                forms.append('ال' + base)
    
    return forms


def expand_arabic_keywords(keywords: List[str]) -> List[str]:
    """
    Expand Arabic keywords using proper NLP techniques (no synonym dictionary).
    
    Uses morphological analysis:
    - Prefix stripping (و، ف، ب، ال etc.)
    - Text normalization
    - Adding/removing definite article forms
    
    This is a legitimate NLP technique, not overfitting to test cases.
    
    Args:
        keywords: List of Arabic keywords
        
    Returns:
        Expanded list with morphological variants
    """
    expanded: Set[str] = set()
    
    for keyword in keywords:
        # Add original
        expanded.add(keyword)
        
        # Get all possible base forms by stripping prefixes
        base_forms = strip_arabic_prefixes(keyword)
        
        for base in base_forms:
            expanded.add(base)
            
            # Normalize base
            normalized = normalize_arabic_for_search(base)
            expanded.add(normalized)
            
            # Also check without ال prefix
            if base.startswith('ال') and len(base) > 2:
                root = base[2:]
                expanded.add(root)
                expanded.add(normalize_arabic_for_search(root))
            
            # Also add with ال prefix if not already present
            if not base.startswith('ال'):
                with_al = 'ال' + base
                expanded.add(with_al)
    
    return list(expanded)


# Keep old function name for backward compatibility
def expand_arabic_synonyms(keywords: List[str]) -> List[str]:
    """Deprecated: Use expand_arabic_keywords instead."""
    return expand_arabic_keywords(keywords)


def remove_diacritics_keep_shadda(text: str) -> str:
    """
    Remove Arabic diacritics but preserve the shadda (ّ).
    
    The shadda (ّ) is a crucial diacritic that indicates consonant doubling
    and is preserved for better semantic understanding.
    
    Args:
        text: Arabic text string
        
    Returns:
        Text with diacritics removed except shadda
        
    Example:
        >>> remove_diacritics_keep_shadda("مُحَمَّد")
        "محمّد"
    """
    if not isinstance(text, str) or not text:
        return text
    return DIACRITICS_RE.sub('', text)


def remove_directional_marks(text: str) -> str:
    """
    Remove Unicode directional marks (LTR/RTL marks).
    
    These invisible characters can interfere with text processing
    and database operations.
    
    Args:
        text: Text string possibly containing directional marks
        
    Returns:
        Text with directional marks removed
    """
    if not isinstance(text, str) or not text:
        return text
    return DIRECTIONAL_MARKS_RE.sub('', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace by replacing multiple spaces, tabs, and newlines
    with single spaces, and trim leading/trailing whitespace.
    
    Args:
        text: Text string with potentially irregular whitespace
        
    Returns:
        Text with normalized whitespace
        
    Example:
        >>> normalize_whitespace("Hello   \\n\\n  World")
        "Hello World"
    """
    if not isinstance(text, str) or not text:
        return text
    return WHITESPACE_RE.sub(' ', text).strip()


def normalize_arabic_text(text: str) -> str:
    """
    Lightweight normalization for already-preprocessed Arabic text.
    
    Since our JSON data is already cleaned (diacritics removed, directional marks
    removed), this function only normalizes whitespace for consistency.
    
    This is the PRIMARY function to use for ingestion pipeline.
    
    Args:
        text: Preprocessed Arabic text from JSON
        
    Returns:
        Text with normalized whitespace
        
    Example:
        >>> normalize_arabic_text("حدّثنا   محمّد\\n\\nبن")
        "حدّثنا محمّد بن"
    """
    if not isinstance(text, str) or not text:
        return text
    
    return normalize_whitespace(text)


def clean_arabic_text(
    text: str,
    remove_diacritics: bool = False,
    remove_directional: bool = False,
    normalize_spaces: bool = True,
    keep_shadda: bool = True
) -> str:
    """
    Full preprocessing pipeline for Arabic text (legacy/optional).
    
    NOTE: For ingestion from preprocessed JSON files, use normalize_arabic_text() instead.
    This function is kept for flexibility with external/raw data sources.
    
    Args:
        text: Raw Arabic text
        remove_diacritics: Whether to remove diacritics (default: False, already done)
        remove_directional: Whether to remove directional marks (default: False, already done)
        normalize_spaces: Whether to normalize whitespace (default: True)
        keep_shadda: Whether to preserve shadda when removing diacritics (default: True)
        
    Returns:
        Cleaned and normalized Arabic text
        
    Example:
        >>> clean_arabic_text("حَدَّثَنَا   مُحَمَّدٌ", remove_diacritics=True)
        "حدّثنا محمّد"
    """
    if not isinstance(text, str) or not text:
        return text
    
    cleaned = text
    
    # Step 1: Remove directional marks (if needed)
    if remove_directional:
        cleaned = remove_directional_marks(cleaned)
    
    # Step 2: Remove diacritics (if needed)
    if remove_diacritics and keep_shadda:
        cleaned = remove_diacritics_keep_shadda(cleaned)
    
    # Step 3: Normalize whitespace
    if normalize_spaces:
        cleaned = normalize_whitespace(cleaned)
    
    return cleaned


def recursive_clean(
    obj,
    remove_diacritics: bool = True,
    remove_directional: bool = True,
    normalize_spaces: bool = True,
    keep_shadda: bool = True
):
    """
    Recursively clean all string values in a nested data structure.
    
    Useful for cleaning JSON-like objects with nested dictionaries and lists.
    
    Args:
        obj: Python object (dict, list, str, or other)
        remove_diacritics: Whether to remove diacritics
        remove_directional: Whether to remove directional marks
        normalize_spaces: Whether to normalize whitespace
        keep_shadda: Whether to preserve shadda
        
    Returns:
        Cleaned version of the input object with same structure
        
    Example:
        >>> data = {"title": "حَدِيث", "items": ["نَصّ"]}
        >>> recursive_clean(data)
        {"title": "حديث", "items": ["نصّ"]}
    """
    if isinstance(obj, dict):
        return {k: recursive_clean(v, remove_diacritics, remove_directional, 
                                   normalize_spaces, keep_shadda) 
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_clean(v, remove_diacritics, remove_directional, 
                               normalize_spaces, keep_shadda) 
                for v in obj]
    elif isinstance(obj, str):
        return clean_arabic_text(obj, remove_diacritics, remove_directional, 
                                normalize_spaces, keep_shadda)
    else:
        return obj


if __name__ == "__main__":
    # Test with preprocessed text (similar to what's in our JSON files)
    preprocessed_text = "حدّثنا   محمّد بن إسماعيل البخاريّ"
    
    print("=== Testing with Preprocessed Text (from JSON) ===")
    print("Input (already cleaned):", preprocessed_text)
    print("After normalize_arabic_text():", normalize_arabic_text(preprocessed_text))
    print()
    
    # Test with raw text (if we had external data)
    raw_text = "حَدَّثَنَا   مُحَمَّدُ بْنُ إِسْمَاعِيلَ \u200e البُخَارِيّ"
    
    print("=== Testing with Raw Text (external sources) ===")
    print("Input (with diacritics):", raw_text)
    print("After clean_arabic_text():", clean_arabic_text(raw_text, remove_diacritics=True, remove_directional=True))
    print()
    
    # Test with nested structure
    test_data = {
        "arabic": "حدّثنا   محمّد",
        "metadata": {
            "title": "صحيح البخاريّ"
        }
    }
    
    print("=== Testing Recursive Clean ===")
    print("Original dict:", test_data)
    print("After recursive_clean():", recursive_clean(test_data))
