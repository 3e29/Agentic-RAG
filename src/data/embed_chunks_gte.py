"""
Hadith Embedding Script - GTE Version (Test with First 5 Batches)

This script uses the Alibaba-NLP/gte-multilingual-base model for embeddings.
Stores in a SEPARATE folder (data/chroma_db_gte) to not mix with E5 embeddings.

Model: Alibaba-NLP/gte-multilingual-base
- 768 embedding dimensions (vs E5's 1024)
- 8192 max tokens (vs E5's 512)
- NO prefix required (unlike E5's "passage:")

Test Mode: Only embeds first 5 batches (250 chunks) for testing.

Input: 
  - ./data/chunks/bukhari_chunks.jsonl
  - ./data/chunks/muslim_chunks.jsonl
  - ./data/raw/bukhari.json (for chapter titles)
  - ./data/raw/muslim.json (for chapter titles)
  
Output: ChromaDB at ./data/chroma_db_gte (SEPARATE from E5 embeddings)

Usage:
    python src/data/embed_chunks_gte.py
"""

import json
import sys
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import httpx
import chromadb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration - GTE Model
# =============================================================================

# GTE Endpoint (different from E5)
MODAL_EMBED_URL = "https://sazaitet110--gte-multilingual-embeddings-embed.modal.run"

# SEPARATE ChromaDB path to not mix with E5 embeddings
CHROMA_DB_PATH = project_root / "data" / "chroma_db_gte"

# Chunk files (JSONL)
CHUNK_FILES = {
    'bukhari': project_root / "data" / "chunks" / "bukhari_chunks.jsonl",
    'muslim': project_root / "data" / "chunks" / "muslim_chunks.jsonl",
}

# Source files for chapter titles (JSON)
SOURCE_FILES = {
    'bukhari': project_root / "data" / "raw" / "bukhari.json",
    'muslim': project_root / "data" / "raw" / "muslim.json",
}

BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 2

# TEST MODE: Only process first N batches
MAX_BATCHES = 5  # 5 batches × 50 = 250 chunks for testing
TEST_MODE = True

# =============================================================================
# Chapter Mapping Loader
# =============================================================================

def load_chapter_mapping(source_path: Path) -> Dict[int, Dict[str, str]]:
    """
    Load chapter ID -> title mapping from source JSON.
    
    Returns:
        Dict mapping chapter_id to {
            "title_en": "The Book of Faith",
            "title_ar": "كتاب الإيمان"
        }
    """
    if not source_path.exists():
        logger.warning(f"Source file not found: {source_path}")
        return {}
    
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        mapping = {}
        for chapter in data.get("chapters", []):
            chapter_id = chapter.get("id")
            if chapter_id is not None:
                mapping[chapter_id] = {
                    "title_en": chapter.get("english", ""),
                    "title_ar": chapter.get("arabic", ""),
                }
        
        logger.info(f"Loaded {len(mapping)} chapters from {source_path.name}")
        return mapping
        
    except Exception as e:
        logger.error(f"Failed to load chapter mapping: {e}")
        return {}


# =============================================================================
# GTE Embedding Pipeline
# =============================================================================

class GTEEmbeddingPipeline:
    """Embedding pipeline using GTE multilingual model."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
        }
        
        # Initialize ChromaDB in SEPARATE folder
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        
        # Get or create collection with cosine similarity
        # Use different collection name suffix to distinguish from E5
        self.collection = self.client.get_or_create_collection(
            name=f"hadith_{collection_name}_gte",
            metadata={
                "hnsw:space": "cosine",
                "description": f"{collection_name.title()} hadiths with GTE embeddings",
                "embedding_model": "Alibaba-NLP/gte-multilingual-base",
                "embedding_dimension": "768",
                "enrichment": "chapter_titles_prepended",
            }
        )
        
        initial_count = self.collection.count()
        logger.info(f"Collection 'hadith_{collection_name}_gte' ready (current: {initial_count} docs)")
        
        # Load chapter mapping
        source_file = SOURCE_FILES.get(collection_name)
        self.chapter_map = load_chapter_mapping(source_file) if source_file else {}
    
    def enrich_text(self, chunk: Dict[str, Any]) -> str:
        """
        Enrich chunk text with chapter context for better semantic search.
        
        Format:
            [Book: The Book of Zakat | كتاب الزكاة]
            <original hadith text>
        """
        chapter_id = chunk.get("chapter_id")
        chapter_info = self.chapter_map.get(chapter_id, {})
        
        title_en = chapter_info.get("title_en", "")
        title_ar = chapter_info.get("title_ar", "")
        
        original_text = chunk.get("text", "")
        
        if title_en or title_ar:
            # Prepend chapter context
            context = f"[Book: {title_en} | {title_ar}]"
            return f"{context}\n{original_text}"
        
        return original_text
    
    def prepare_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare full metadata for ChromaDB storage.
        """
        chapter_id = chunk.get("chapter_id")
        chapter_info = self.chapter_map.get(chapter_id, {})
        
        return {
            # Original chunk metadata
            'collection': chunk.get('collection', self.collection_name),
            'language': chunk.get('language', ''),
            'book_id': int(chunk.get('book_id', 0)),
            'chapter_id': int(chunk.get('chapter_id', 0)),
            'hadith_id': int(chunk.get('hadith_id', 0)),
            'hadith_id_in_book': int(chunk.get('hadith_id_in_book', 0)),
            'is_chunked': bool(chunk.get('is_chunked', False)),
            'chunk_index': int(chunk.get('chunk_index', 0)),
            'total_chunks': int(chunk.get('total_chunks', 1)),
            'narrator': chunk.get('narrator', ''),
            'parent_hadith_id': chunk.get('parent_hadith_id', ''),
            
            # Enriched chapter metadata
            'chapter_title_en': chapter_info.get('title_en', ''),
            'chapter_title_ar': chapter_info.get('title_ar', ''),
            
            # Mark as GTE embedding
            'embedding_model': 'gte-multilingual-base',
        }
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        client: httpx.AsyncClient
    ) -> Optional[List[List[float]]]:
        """Generate embeddings via Modal GTE API with retry logic."""
        
        # GTE does NOT need any prefix - send raw texts
        # This is different from E5 which needs "passage:" prefix
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    MODAL_EMBED_URL,
                    json={"texts": texts},  # No prefix needed for GTE
                    timeout=120.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Handle response format
                if isinstance(result, dict):
                    embeddings = result.get('embeddings', result.get('data', []))
                else:
                    embeddings = result
                
                if not embeddings or len(embeddings) != len(texts):
                    raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings) if embeddings else 0}")
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"GTE embedding attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    return None
        
        return None
    
    async def process_batch(
        self, 
        batch: List[Dict[str, Any]], 
        batch_num: int,
        client: httpx.AsyncClient
    ):
        """Process a single batch of chunks."""
        
        # Prepare data
        ids = [chunk['chunk_id'] for chunk in batch]
        enriched_texts = [self.enrich_text(chunk) for chunk in batch]
        original_texts = [chunk['text'] for chunk in batch]
        metadatas = [self.prepare_metadata(chunk) for chunk in batch]
        
        # Generate embeddings for ENRICHED texts using GTE
        embeddings = await self.generate_embeddings(enriched_texts, client)
        
        if embeddings is None:
            logger.error(f"Batch {batch_num}: Failed to generate GTE embeddings")
            self.stats['failed'] += len(batch)
            return
        
        try:
            # Upsert to ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=original_texts,
                metadatas=metadatas,
            )
            
            self.stats['processed'] += len(batch)
            logger.info(
                f"Batch {batch_num}: {len(batch)} chunks | "
                f"Total: {self.stats['processed']}/{self.stats['total_chunks']}"
            )
            
        except Exception as e:
            logger.error(f"Batch {batch_num}: Storage failed - {e}")
            self.stats['failed'] += len(batch)
    
    async def run(self, chunk_file: Path):
        """Run the GTE embedding pipeline."""
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info(f"GTE EMBEDDING PIPELINE: {self.collection_name.upper()}")
        logger.info("=" * 70)
        logger.info(f"Model: Alibaba-NLP/gte-multilingual-base")
        logger.info(f"Embedding dimension: 768")
        logger.info(f"Chunk file: {chunk_file}")
        logger.info(f"Chapter map: {len(self.chapter_map)} chapters loaded")
        logger.info(f"Batch size: {BATCH_SIZE}")
        if TEST_MODE:
            logger.info(f"⚠️  TEST MODE: Only processing first {MAX_BATCHES} batches ({MAX_BATCHES * BATCH_SIZE} chunks)")
        logger.info("=" * 70)
        
        # Load chunks (limited in test mode)
        chunks = []
        max_chunks = MAX_BATCHES * BATCH_SIZE if TEST_MODE else float('inf')
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if len(chunks) >= max_chunks:
                    break
                try:
                    if line.strip():
                        chunk = json.loads(line.strip())
                        chunks.append(chunk)
                        self.stats['total_chunks'] += 1
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at line {line_num}: {e}")
                    self.stats['failed'] += 1
        
        logger.info(f"Loaded {len(chunks)} chunks from file")
        
        # Process in batches
        async with httpx.AsyncClient() as client:
            batch_count = 0
            for i in range(0, len(chunks), BATCH_SIZE):
                if TEST_MODE and batch_count >= MAX_BATCHES:
                    break
                    
                batch = chunks[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                await self.process_batch(batch, batch_num, client)
                batch_count += 1
        
        # Summary
        duration = time.time() - start_time
        final_count = self.collection.count()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"{self.collection_name.upper()} GTE EMBEDDING COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Total chunks: {self.stats['total_chunks']}")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"ChromaDB total: {final_count} documents")
        logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        if self.stats['processed'] > 0:
            logger.info(f"Speed: {self.stats['processed']/duration:.1f} chunks/sec")
        logger.info(f"Output: {CHROMA_DB_PATH}")
        logger.info("=" * 70)
        
        return self.stats


# =============================================================================
# Main
# =============================================================================

async def main():
    """Main entry point."""
    logger.info("\n" + "=" * 70)
    logger.info("GTE HADITH EMBEDDING - Test Mode (First 5 Batches)")
    logger.info("Model: Alibaba-NLP/gte-multilingual-base")
    logger.info("Output: data/chroma_db_gte (SEPARATE from E5)")
    logger.info("=" * 70)
    
    all_stats = []
    
    for collection_name, chunk_file in CHUNK_FILES.items():
        if not chunk_file.exists():
            logger.warning(f"Chunk file not found: {chunk_file}. Skipping {collection_name}.")
            continue
        
        pipeline = GTEEmbeddingPipeline(collection_name)
        
        try:
            stats = await pipeline.run(chunk_file)
            all_stats.append({
                'collection': collection_name,
                **stats
            })
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.error(f"Pipeline failed for {collection_name}: {e}", exc_info=True)
            continue
    
    # Overall summary
    if all_stats:
        logger.info("\n" + "=" * 70)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 70)
        
        total = sum(s['total_chunks'] for s in all_stats)
        processed = sum(s['processed'] for s in all_stats)
        failed = sum(s['failed'] for s in all_stats)
        
        logger.info(f"Total: {total} | Processed: {processed} | Failed: {failed}")
        logger.info(f"\nGTE Embeddings stored in: {CHROMA_DB_PATH}")
        logger.info("Collections created:")
        for s in all_stats:
            logger.info(f"  - hadith_{s['collection']}_gte")
        logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
