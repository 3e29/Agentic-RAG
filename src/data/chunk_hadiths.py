"""
Hadith Chunking Script - Phase 1: Chunking with LangChain

This script processes hadith collections one at a time, chunking them using
LangChain's RecursiveCharacterTextSplitter for both Arabic and English texts.

Output:
- data/chunks/bukhari_chunks.jsonl
- data/chunks/muslim_chunks.jsonl

Each line contains a chunk with its metadata, ready for Phase 2 embedding.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# LangChain text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.arabic_processing import normalize_arabic_text


# Configuration
RAW_DATA_PATHS = {
    'bukhari': project_root / "data" / "raw" / "bukhari.json",
    'muslim': project_root / "data" / "raw" / "muslim.json"
}

OUTPUT_DIR = project_root / "data" / "chunks"
OUTPUT_FILES = {
    'bukhari': OUTPUT_DIR / "bukhari_chunks.jsonl",
    'muslim': OUTPUT_DIR / "muslim_chunks.jsonl"
}

# Chunking parameters (optimized for multilingual-e5-large: 512 token limit)
# ~800 chars ≈ 400 tokens (conservative estimate for Arabic/English)
MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 100  # Character overlap for context continuity

# Arabic separators for RecursiveCharacterTextSplitter
ARABIC_SEPARATORS = [
    "\n\n",  # Double newline (paragraphs)
    "\n",    # Single newline
    ".",     # Period
    "۔",     # Urdu/Arabic period
    "؟",     # Arabic question mark
    "!",     # Exclamation
    "؛",     # Arabic semicolon
    ":",     # Colon
    "،",     # Arabic comma
    ",",     # Comma
    " ",     # Space
    ""       # Character-level fallback
]

# English separators
ENGLISH_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ": ",
    ", ",
    " ",
    ""
]


def create_text_splitter(language: str) -> RecursiveCharacterTextSplitter:
    """
    Create a LangChain text splitter for the specified language.
    
    Args:
        language: 'arabic' or 'english'
        
    Returns:
        RecursiveCharacterTextSplitter configured for the language
    """
    separators = ARABIC_SEPARATORS if language == 'arabic' else ENGLISH_SEPARATORS
    
    return RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=separators,
        is_separator_regex=False
    )


def generate_parent_id(collection: str, book_id: int, hadith_id: int) -> str:
    """
    Generate a unique parent ID for a hadith.
    
    Format: {collection}_{book_id}_{hadith_id}
    Example: sahih_al-bukhari_1_1
    """
    collection_slug = collection.lower().replace(' ', '_')
    return f"{collection_slug}_{book_id}_{hadith_id}"


def chunk_hadith(
    hadith: Dict[str, Any],
    collection: str,
    book_id: int,
    chapter_id: int,
    book_number: str,
    chapter_number: str,
    chapter_title_en: str = "",
    chapter_title_ar: str = "",
) -> List[Dict[str, Any]]:
    """
    Chunk a single hadith into Arabic and English chunks using LangChain.
    
    Args:
        hadith: Hadith dictionary with 'arabic', 'english', 'narrator', 'id', 'idInBook'
        collection: Collection name (e.g., "Sahih al-Bukhari")
        book_id: Book ID
        chapter_id: Chapter ID
        book_number: Book number (e.g., "1")
        chapter_number: Chapter number (e.g., "1")
        chapter_title_en: English chapter title (for BM25 filtering)
        chapter_title_ar: Arabic chapter title (for BM25 filtering)
        
    Returns:
        List of chunk dictionaries ready for embedding
    """
    chunks = []
    
    hadith_id = hadith.get('id', 0)
    hadith_id_in_book = hadith.get('idInBook', 0)
    narrator = hadith.get('narrator', '')
    
    # Generate parent ID for linking chunks back to source
    parent_id = generate_parent_id(collection, book_id, hadith_id)
    
    # Base metadata shared by all chunks
    base_metadata = {
        'collection': collection,
        'book_id': book_id,
        'chapter_id': chapter_id,
        'book_number': book_number,
        'chapter_number': chapter_number,
        'chapter_title_en': chapter_title_en,  # Added for BM25
        'chapter_title_ar': chapter_title_ar,  # Added for BM25
        'hadith_id': hadith_id,
        'hadith_id_in_book': hadith_id_in_book,
        'narrator': narrator,
        'parent_hadith_id': parent_id
    }
    
    # Process Arabic text
    arabic_text = hadith.get('arabic', '').strip()
    if arabic_text:
        # Normalize Arabic text
        arabic_text = normalize_arabic_text(arabic_text)
        
        # Create splitter for Arabic
        arabic_splitter = create_text_splitter('arabic')
        
        # Split text
        arabic_chunks = arabic_splitter.split_text(arabic_text)
        
        # Create chunk objects
        for idx, chunk_text in enumerate(arabic_chunks):
            chunk_id = f"{parent_id}_arabic_chunk_{idx}"
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'language': 'arabic',
                'chunk_index': idx,
                'total_chunks': len(arabic_chunks),
                'chunk_size': len(chunk_text),
                'is_chunked': len(arabic_chunks) > 1,
                **base_metadata
            })
    
    # Process English text
    english_text = hadith.get('english', '').strip()
    if english_text:
        # Create splitter for English
        english_splitter = create_text_splitter('english')
        
        # Split text
        english_chunks = english_splitter.split_text(english_text)
        
        # Create chunk objects
        for idx, chunk_text in enumerate(english_chunks):
            chunk_id = f"{parent_id}_english_chunk_{idx}"
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'language': 'english',
                'chunk_index': idx,
                'total_chunks': len(english_chunks),
                'chunk_size': len(chunk_text),
                'is_chunked': len(english_chunks) > 1,
                **base_metadata
            })
    
    return chunks


def process_collection(collection_name: str, data_path: Path, output_file: Path) -> Dict[str, Any]:
    """
    Process a single hadith collection one hadith at a time.
    
    Args:
        collection_name: Name of collection ('bukhari' or 'muslim')
        data_path: Path to JSON data file
        output_file: Path to output JSONL file
        
    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*70}")
    print(f"Processing: {collection_name.title()}")
    print(f"Input: {data_path}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")
    
    # Load collection data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    collection_english = data['metadata']['english']['title']
    chapters_data = data.get('chapters', [])
    hadiths_data = data.get('hadiths', [])
    
    # Statistics
    total_hadiths = 0
    total_chunks = 0
    chunked_hadiths = 0
    
    start_time = datetime.now()
    
    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Process hadiths directly (they're in a flat array)
        for hadith in hadiths_data:
            total_hadiths += 1
            
            # Extract metadata
            hadith_id = hadith.get('id', 0)
            chapter_id = hadith.get('chapterId', 0)
            book_id = hadith.get('bookId', 1)
            
            # Find chapter info and extract titles
            chapter_info = next((c for c in chapters_data if c['id'] == chapter_id), {})
            book_number = str(book_id)
            chapter_number = str(chapter_id)
            chapter_title_en = chapter_info.get('english', '')
            chapter_title_ar = chapter_info.get('arabic', '')
            
            # Extract text from nested structure
            hadith_text = {
                'arabic': hadith.get('arabic', ''),
                'english': hadith.get('english', {}).get('text', '') if isinstance(hadith.get('english'), dict) else hadith.get('english', ''),
                'narrator': hadith.get('english', {}).get('narrator', '') if isinstance(hadith.get('english'), dict) else '',
                'id': hadith_id,
                'idInBook': hadith.get('idInBook', hadith_id)
            }
            
            # Chunk this single hadith
            chunks = chunk_hadith(
                hadith=hadith_text,
                collection=collection_english,
                book_id=book_id,
                chapter_id=chapter_id,
                book_number=book_number,
                chapter_number=chapter_number,
                chapter_title_en=chapter_title_en,
                chapter_title_ar=chapter_title_ar,
            )
            
            # Write chunks to output file immediately
            for chunk in chunks:
                out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total_chunks += 1
            
            # Track if hadith was chunked
            if any(c['is_chunked'] for c in chunks):
                chunked_hadiths += 1
            
            # Progress indicator (every 100 hadiths)
            if total_hadiths % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_hadiths / elapsed if elapsed > 0 else 0
                print(f"  Processed: {total_hadiths} hadiths, {total_chunks} chunks "
                      f"({rate:.1f} hadiths/sec)")
    
    # Final statistics
    elapsed = (datetime.now() - start_time).total_seconds()
    
    stats = {
        'collection': collection_name,
        'total_hadiths': total_hadiths,
        'total_chunks': total_chunks,
        'chunked_hadiths': chunked_hadiths,
        'unchunked_hadiths': total_hadiths - chunked_hadiths,
        'avg_chunks_per_hadith': total_chunks / total_hadiths if total_hadiths > 0 else 0,
        'processing_time_seconds': elapsed,
        'hadiths_per_second': total_hadiths / elapsed if elapsed > 0 else 0
    }
    
    print(f"\n{collection_name.title()} Statistics:")
    print(f"  Total Hadiths: {stats['total_hadiths']}")
    print(f"  Total Chunks: {stats['total_chunks']}")
    print(f"  Chunked Hadiths: {stats['chunked_hadiths']}")
    print(f"  Unchunked Hadiths: {stats['unchunked_hadiths']}")
    print(f"  Avg Chunks/Hadith: {stats['avg_chunks_per_hadith']:.2f}")
    print(f"  Processing Time: {stats['processing_time_seconds']:.2f}s")
    print(f"  Rate: {stats['hadiths_per_second']:.1f} hadiths/sec")
    
    return stats


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("HADITH CHUNKING - Phase 1")
    print("Using LangChain RecursiveCharacterTextSplitter")
    print("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each collection
    all_stats = []
    
    for collection_name, data_path in RAW_DATA_PATHS.items():
        if not data_path.exists():
            print(f"\nWarning: {data_path} not found. Skipping {collection_name}.")
            continue
        
        output_file = OUTPUT_FILES[collection_name]
        stats = process_collection(collection_name, data_path, output_file)
        all_stats.append(stats)
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    total_hadiths = sum(s['total_hadiths'] for s in all_stats)
    total_chunks = sum(s['total_chunks'] for s in all_stats)
    total_time = sum(s['processing_time_seconds'] for s in all_stats)
    
    print(f"\nTotal Hadiths Processed: {total_hadiths}")
    print(f"Total Chunks Generated: {total_chunks}")
    print(f"Total Processing Time: {total_time:.2f}s")
    print(f"Overall Rate: {total_hadiths / total_time:.1f} hadiths/sec")
    
    print(f"\nOutput Files:")
    for collection_name, output_file in OUTPUT_FILES.items():
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  {collection_name}: {output_file} ({file_size:.2f} MB)")
    
    print("\nPhase 1 Complete! Ready for Phase 2 (Embedding).")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
