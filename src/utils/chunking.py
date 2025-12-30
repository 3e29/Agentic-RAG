"""
Semantic Chunking Utility for Arabic Hadith Text

This module implements sentence-based semantic chunking for Arabic text,
ensuring that long hadiths are split into manageable chunks while preserving
context and maintaining full citation metadata.

Key Features:
- Sentence boundary detection using Arabic punctuation
- Overlap between chunks for context continuity
- Metadata preservation for complete citations
- No LLM required - fast and deterministic

Usage:
    from src.utils.chunking import chunk_hadith
    
    chunks = chunk_hadith(hadith_text, metadata, max_chunk_size=800)
"""

import re
from typing import List, Dict, Any


# Arabic sentence boundary patterns
ARABIC_SENTENCE_BOUNDARIES = re.compile(
    r'[.۰٠؟؛]|(?<=[،,;:])\s+',  # Period, Arabic punctuation, or after comma with space
    flags=re.UNICODE
)

# More aggressive split pattern for very long segments
SECONDARY_SPLIT_PATTERN = re.compile(
    r'[،,;:]',  # Any comma or semicolon
    flags=re.UNICODE
)


def split_into_sentences(text: str, language: str = 'arabic') -> List[str]:
    """
    Split text into sentences using punctuation boundaries.
    
    Args:
        text: Text to split
        language: 'arabic' or 'english' for proper sentence splitting
        
    Returns:
        List of sentence strings
    """
    if not text:
        return []
    
    if language == 'english':
        # English sentence boundaries: periods, exclamation, question marks
        import re
        pattern = re.compile(r'[.!?]+\s+', flags=re.UNICODE)
        sentences = pattern.split(text)
    else:
        # Arabic sentence boundaries (default)
        sentences = ARABIC_SENTENCE_BOUNDARIES.split(text)
    
    # Clean up: remove empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def chunk_text_with_overlap(
    sentences: List[str],
    max_chunk_size: int = 800,
    overlap_sentences: int = 1
) -> List[str]:
    """
    Group sentences into chunks with overlap between consecutive chunks.
    
    Args:
        sentences: List of sentence strings
        max_chunk_size: Maximum characters per chunk (default: 800)
        overlap_sentences: Number of sentences to overlap (default: 1)
        
    Returns:
        List of text chunks
    """
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_len = len(sentence)
        
        # If single sentence exceeds max_chunk_size, split it further
        if sentence_len > max_chunk_size and not current_chunk:
            # Split by secondary patterns (commas)
            sub_parts = SECONDARY_SPLIT_PATTERN.split(sentence)
            sub_parts = [p.strip() for p in sub_parts if p.strip()]
            
            # Group sub-parts
            for part in sub_parts:
                if len(part) > max_chunk_size:
                    # Force split at max_chunk_size as last resort
                    for j in range(0, len(part), max_chunk_size):
                        chunks.append(part[j:j+max_chunk_size])
                else:
                    if current_size + len(part) + 2 <= max_chunk_size:
                        current_chunk.append(part)
                        current_size += len(part) + 2  # +2 for space and punctuation
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk) + '.')
                        current_chunk = [part]
                        current_size = len(part)
            
            if current_chunk:
                chunks.append(' '.join(current_chunk) + '.')
                current_chunk = []
                current_size = 0
            
            i += 1
            continue
        
        # Normal case: add sentence to current chunk
        if current_size + sentence_len + 2 <= max_chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_len + 2  # +2 for space/punctuation
            i += 1
        else:
            # Current chunk is full, save it
            if current_chunk:
                chunks.append(' '.join(current_chunk) + '.')
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - overlap_sentences)
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(s) for s in current_chunk) + len(current_chunk) * 2
            else:
                # Edge case: single sentence too long
                i += 1
    
    # Add remaining sentences
    if current_chunk:
        chunks.append(' '.join(current_chunk) + '.')
    
    return chunks


def chunk_hadith(
    hadith_text: str,
    metadata: Dict[str, Any],
    max_chunk_size: int = 800,
    overlap_sentences: int = 1,
    min_chunk_size: int = 50,
    language: str = 'arabic'
) -> List[Dict[str, Any]]:
    """
    Chunk a hadith text into semantic chunks with metadata preservation.
    
    Each chunk maintains full citation information for traceability.
    Short hadiths (<= max_chunk_size) are kept whole.
    
    Args:
        hadith_text: The hadith text to chunk
        metadata: Dictionary containing hadith metadata (collection, book, chapter, etc.)
        max_chunk_size: Maximum characters per chunk (default: 800 ≈ 400 tokens)
        overlap_sentences: Number of sentences to overlap between chunks (default: 1)
        min_chunk_size: Minimum chunk size to keep (default: 50)
        language: 'arabic' or 'english' for proper sentence splitting
        
    Returns:
        List of chunk dictionaries, each containing:
        - text: The chunk text
        - metadata: Full metadata with chunk info including parent_hadith_id
        
    Example:
        >>> metadata = {
        ...     'hadith_id': 1,
        ...     'collection_english': 'Sahih al-Bukhari',
        ...     'book_id': 1,
        ...     'chapter_id': 1
        ... }
        >>> chunks = chunk_hadith(long_hadith_text, metadata, language='arabic')
        >>> len(chunks)
        3
        >>> chunks[0]['metadata']['parent_hadith_id']
        'sahih_al-bukhari_1_1'
    """
    # Generate parent hadith ID for linking all chunks
    parent_id = f"{metadata['collection_english'].lower().replace(' ', '_')}_{metadata['book_id']}_{metadata['hadith_id']}"
    
    # Short hadiths: no chunking needed
    if len(hadith_text) <= max_chunk_size:
        return [{
            'text': hadith_text,
            'metadata': {
                **metadata,
                'is_chunked': False,
                'chunk_index': 0,
                'total_chunks': 1,
                'chunk_size': len(hadith_text),
                'parent_hadith_id': parent_id,
                'language': language
            }
        }]
    
    # Long hadiths: apply chunking
    sentences = split_into_sentences(hadith_text, language=language)
    
    # If splitting didn't work well, fall back to simple split
    if len(sentences) <= 1:
        # Force split at max_chunk_size
        text_chunks = []
        for i in range(0, len(hadith_text), max_chunk_size):
            chunk = hadith_text[i:i+max_chunk_size]
            if len(chunk) >= min_chunk_size:
                text_chunks.append(chunk)
    else:
        text_chunks = chunk_text_with_overlap(sentences, max_chunk_size, overlap_sentences)
    
    # Filter out very small chunks
    text_chunks = [c for c in text_chunks if len(c) >= min_chunk_size]
    
    # Create chunk objects with metadata
    chunks = []
    total_chunks = len(text_chunks)
    
    for idx, chunk_text in enumerate(text_chunks):
        chunk_obj = {
            'text': chunk_text,
            'metadata': {
                **metadata,
                'is_chunked': True,
                'chunk_index': idx,
                'total_chunks': total_chunks,
                'chunk_size': len(chunk_text),
                'parent_hadith_id': parent_id,
                'language': language
            }
        }
        chunks.append(chunk_obj)
    
    return chunks


def get_chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'min_chunk_size': 0,
            'max_chunk_size': 0,
            'chunked_count': 0,
            'unchunked_count': 0
        }
    
    sizes = [c['metadata']['chunk_size'] for c in chunks]
    chunked_count = sum(1 for c in chunks if c['metadata'].get('is_chunked', False))
    
    return {
        'total_chunks': len(chunks),
        'avg_chunk_size': sum(sizes) / len(sizes),
        'min_chunk_size': min(sizes),
        'max_chunk_size': max(sizes),
        'chunked_count': chunked_count,
        'unchunked_count': len(chunks) - chunked_count
    }


if __name__ == "__main__":
    # Test with sample hadith
    print("Testing Arabic Hadith Chunking\n" + "="*70)
    
    # Short hadith (no chunking needed)
    short_hadith = "حدثنا محمد بن إسماعيل. قال النبي صلى الله عليه وسلم."
    short_meta = {'hadith_id': 1, 'collection': 'Test'}
    
    short_chunks = chunk_hadith(short_hadith, short_meta)
    print(f"\nShort hadith ({len(short_hadith)} chars):")
    print(f"  Chunks: {len(short_chunks)}")
    print(f"  Chunked: {short_chunks[0]['metadata']['is_chunked']}")
    
    # Long hadith (needs chunking)
    long_hadith = "حدثنا محمد بن إسماعيل البخاري. " * 50  # Repeat to make it long
    long_meta = {'hadith_id': 2, 'collection': 'Test', 'book_id': 1}
    
    long_chunks = chunk_hadith(long_hadith, long_meta, max_chunk_size=800)
    print(f"\nLong hadith ({len(long_hadith)} chars):")
    print(f"  Chunks: {len(long_chunks)}")
    print(f"  Chunked: {long_chunks[0]['metadata']['is_chunked']}")
    
    for i, chunk in enumerate(long_chunks):
        print(f"  Chunk {i}: {chunk['metadata']['chunk_size']} chars")
    
    # Statistics
    stats = get_chunk_statistics(long_chunks)
    print(f"\nStatistics:")
    print(f"  Avg chunk size: {stats['avg_chunk_size']:.0f} chars")
    print(f"  Min: {stats['min_chunk_size']}, Max: {stats['max_chunk_size']}")
