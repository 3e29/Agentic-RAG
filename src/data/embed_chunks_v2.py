"""
Hadith Embedding Script v2 - With Chapter Context Enrichment

This enhanced script reads chunked hadiths, enriches them with chapter titles
from the source JSON files, and stores embeddings in ChromaDB.

Key Improvements over v1:
1. Chapter Context Enrichment - Prepends chapter titles to text for better semantic search
   (e.g., searching "Zakat" finds hadiths in "Book of Zakat" even if text doesn't mention it)
2. Full Metadata Preservation - Stores all chunk metadata for filtering
3. Bilingual Support - Stores both Arabic and English chapter titles
4. Idempotent Operations - Uses upsert() for safe re-runs

Input: 
  - ./data/chunks/bukhari_chunks.jsonl
  - ./data/chunks/muslim_chunks.jsonl
  - ./data/raw/bukhari.json (for chapter titles)
  - ./data/raw/muslim.json (for chapter titles)
  
Output: ChromaDB at ./data/chroma_db

Usage:
    python src/data/embed_chunks_v2.py
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
# Configuration
# =============================================================================

MODAL_EMBED_URL = "https://sazaitet110--multilingual-e5-embeddings-embed.modal.run"
CHROMA_DB_PATH = project_root / "data" / "chroma_db"

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
# Embedding Pipeline
# =============================================================================

class EmbeddingPipelineV2:
    """Enhanced pipeline with chapter context enrichment."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
        }
        
        # Initialize ChromaDB
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        
        # Get or create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=f"hadith_{collection_name}",
            metadata={
                "hnsw:space": "cosine",
                "description": f"{collection_name.title()} hadiths with chapter context",
                "embedding_model": "multilingual-e5-large-instruct",
                "enrichment": "chapter_titles_prepended",
            }
        )
        
        initial_count = self.collection.count()
        logger.info(f"Collection 'hadith_{collection_name}' ready (current: {initial_count} docs)")
        
        # Load chapter mapping
        source_file = SOURCE_FILES.get(collection_name)
        self.chapter_map = load_chapter_mapping(source_file) if source_file else {}
    
    def enrich_text(self, chunk: Dict[str, Any]) -> str:
        """
        Enrich chunk text with chapter context for better semantic search.
        
        Format:
            [Book: The Book of Zakat | كتاب الزكاة]
            <original hadith text>
        
        This allows semantic search for "Zakat" to find ALL hadiths in that chapter,
        even if the individual hadith text doesn't explicitly mention "Zakat".
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
        
        ChromaDB metadata must be str, int, float, or bool.
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
            
            # Enriched chapter metadata (for filtering by chapter title!)
            'chapter_title_en': chapter_info.get('title_en', ''),
            'chapter_title_ar': chapter_info.get('title_ar', ''),
        }
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        client: httpx.AsyncClient
    ) -> Optional[List[List[float]]]:
        """Generate embeddings via Modal API with retry logic."""
        
        # Add "passage:" prefix for E5 model (documents/passages)
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    MODAL_EMBED_URL,
                    json={"texts": prefixed_texts},
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
                logger.warning(f"Embedding attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
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
        
        # Generate embeddings for ENRICHED texts
        embeddings = await self.generate_embeddings(enriched_texts, client)
        
        if embeddings is None:
            logger.error(f"Batch {batch_num}: Failed to generate embeddings")
            self.stats['failed'] += len(batch)
            return
        
        try:
            # Upsert to ChromaDB (store ORIGINAL text as document for display)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=original_texts,  # Store original for retrieval display
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
        """Run the embedding pipeline."""
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info(f"EMBEDDING PIPELINE v2: {self.collection_name.upper()}")
        logger.info("=" * 70)
        logger.info(f"Chunk file: {chunk_file}")
        logger.info(f"Chapter map: {len(self.chapter_map)} chapters loaded")
        logger.info(f"Batch size: {BATCH_SIZE}")
        logger.info("=" * 70)
        
        # Load all chunks
        chunks = []
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
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
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                await self.process_batch(batch, batch_num, client)
        
        # Summary
        duration = time.time() - start_time
        final_count = self.collection.count()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"{self.collection_name.upper()} COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Total chunks: {self.stats['total_chunks']}")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"ChromaDB total: {final_count} documents")
        logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        if self.stats['processed'] > 0:
            logger.info(f"Speed: {self.stats['processed']/duration:.1f} chunks/sec")
        logger.info("=" * 70)
        
        return self.stats


# =============================================================================
# Main
# =============================================================================

async def main():
    """Main entry point."""
    logger.info("\n" + "=" * 70)
    logger.info("HADITH EMBEDDING v2 - With Chapter Context Enrichment")
    logger.info("=" * 70)
    
    all_stats = []
    
    for collection_name, chunk_file in CHUNK_FILES.items():
        if not chunk_file.exists():
            logger.warning(f"Chunk file not found: {chunk_file}. Skipping {collection_name}.")
            continue
        
        pipeline = EmbeddingPipelineV2(collection_name)
        
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
        logger.info("\nEmbedding complete! ChromaDB ready with chapter-enriched vectors.")
        logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
