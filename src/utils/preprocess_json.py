"""
Comprehensive JSON Preprocessing Utility

This module consolidates all preprocessing steps from script.py, script2.py,
and script3.py into a single function for processing Hadith JSON files.

Preprocessing steps:
1. JSON formatting and validation
2. Remove Unicode directional marks (LTR/RTL)
3. Remove Arabic diacritics (keeping shadda)
4. Validate output

Usage:
    python -m src.utils.preprocess_json <input_file> [output_file]
    
Example:
    python -m src.utils.preprocess_json ./data/raw/bukhari.json
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict


# Regex pattern for Arabic diacritics EXCEPT shadda (ّ U+0651)
DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u0650\u0652-\u065F\u0670\u06D6-\u06ED]+',
    flags=re.UNICODE
)

# Regex pattern for directional marks (LTR/RTL)
DIRECTIONAL_MARKS_RE = re.compile(r'[\u200e\u200f]+', flags=re.UNICODE)


def remove_diacritics_keep_shadda(text: str) -> str:
    """
    Remove Arabic diacritics but preserve the shadda (ّ).
    
    Args:
        text: Arabic text string
        
    Returns:
        Text with diacritics removed except shadda
    """
    if not isinstance(text, str):
        return text
    return DIACRITICS_RE.sub('', text)


def remove_directional_marks(text: str) -> str:
    """
    Remove Unicode directional marks (LTR/RTL marks).
    
    Args:
        text: Text string
        
    Returns:
        Text with directional marks removed
    """
    if not isinstance(text, str):
        return text
    return DIRECTIONAL_MARKS_RE.sub('', text)


def recursive_clean(obj: Any) -> Any:
    """
    Recursively clean all strings in a nested data structure.
    
    Applies:
    1. Remove directional marks
    2. Remove diacritics (keep shadda)
    
    Args:
        obj: Python object (dict, list, str, or other)
        
    Returns:
        Cleaned version of the input object
    """
    if isinstance(obj, dict):
        return {k: recursive_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_clean(v) for v in obj]
    elif isinstance(obj, str):
        # Step 1: Remove directional marks
        cleaned = remove_directional_marks(obj)
        # Step 2: Remove diacritics (keep shadda)
        cleaned = remove_diacritics_keep_shadda(cleaned)
        return cleaned
    else:
        return obj


def preprocess_json_file(
    input_path: str,
    output_path: str = None,
    indent: int = 4,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Preprocess a Hadith JSON file with all cleaning steps.
    
    Steps performed:
    1. Load and validate JSON
    2. Remove directional marks from all strings
    3. Remove diacritics (keeping shadda) from all strings
    4. Format with proper indentation
    5. Save to output file
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (default: overwrite input)
        indent: JSON indentation level (default: 4)
        validate: Whether to validate the result (default: True)
        
    Returns:
        Dictionary with processing statistics
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If input is not valid JSON
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    print(f"[LOAD] Loading: {input_path}")
    
    # Step 1: Load JSON
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}")
        raise
    
    print(f"[OK] Loaded {len(json.dumps(data))} characters")
    
    # Step 2: Count marks before cleaning
    stats = {
        'diacritics_before': 0,
        'directional_before': 0,
        'diacritics_after': 0,
        'directional_after': 0
    }
    
    if validate:
        json_str = json.dumps(data, ensure_ascii=False)
        stats['diacritics_before'] = len(DIACRITICS_RE.findall(json_str))
        stats['directional_before'] = len(DIRECTIONAL_MARKS_RE.findall(json_str))
        print(f"[STATS] Before: {stats['diacritics_before']} diacritics, {stats['directional_before']} directional marks")
    
    # Step 3: Clean all strings
    print("[CLEAN] Cleaning text...")
    cleaned_data = recursive_clean(data)
    
    # Step 4: Validate after cleaning
    if validate:
        json_str = json.dumps(cleaned_data, ensure_ascii=False)
        stats['diacritics_after'] = len(DIACRITICS_RE.findall(json_str))
        stats['directional_after'] = len(DIRECTIONAL_MARKS_RE.findall(json_str))
        print(f"[STATS] After: {stats['diacritics_after']} diacritics, {stats['directional_after']} directional marks")
    
    # Step 5: Save to output file
    print(f"[SAVE] Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=indent)
    
    print("[OK] Preprocessing complete!")
    
    # Calculate removed counts
    stats['diacritics_removed'] = stats['diacritics_before'] - stats['diacritics_after']
    stats['directional_removed'] = stats['directional_before'] - stats['directional_after']
    
    return stats


def main():
    """CLI entry point for preprocessing JSON files."""
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.preprocess_json <input_file> [output_file]")
        print("\nExample:")
        print("  python -m src.utils.preprocess_json ./data/raw/bukhari.json")
        print("  python -m src.utils.preprocess_json ./data/raw/muslim.json ./data/processed/muslim_clean.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 80)
    print("JSON Preprocessing Utility")
    print("=" * 80)
    
    try:
        stats = preprocess_json_file(input_file, output_file)
        
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Diacritics removed: {stats['diacritics_removed']}")
        print(f"Directional marks removed: {stats['directional_removed']}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
