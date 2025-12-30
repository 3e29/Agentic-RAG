"""
Hadith Embedding Script for BGE-M3 - With Chapter Context Enrichment

This enhanced script reads chunked hadiths, enriches them with chapter titles
from the source JSON files, and stores MULTI-VECTOR embeddings (dense + sparse)
in ChromaDB using the BGE-M3 model.

Key Features:
1. Chapter Context Enrichment - Prepends chapter titles to text for better semantic search
2. Multi-Vector Storage - Stores both dense (1024-dim) and sparse vectors
3. Full Metadata Preservation - Stores all chunk metadata for filtering
4. Bilingual Support - Stores both Arabic and English chapter titles
5. Idempotent Operations - Uses upsert() for safe re-runs

Model: BAAI/bge-m3
- 1024 embedding dimensions (dense)
- Sparse vectors for keyword matching (learned BM25-like)
- 8192 max tokens
- Best Arabic MIRACL benchmark performance
- NO prefix required

Input: 
  - ./data/chunks/bukhari_chunks.jsonl
  - ./data/chunks/muslim_chunks.jsonl
  - ./data/raw/bukhari.json (for chapter titles)
  - ./data/raw/muslim.json (for chapter titles)
  
Output: ChromaDB at ./data/chroma_db_bge_m3

Usage:
    python src/data/embed_chunks_bge_m3.py
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

# BGE-M3 Multi-Vector Endpoint (dense + sparse in one call)
MODAL_EMBED_URL = "https://alaapocket3--bge-m3-embeddings-embed-multi-endpoint.modal.run"

# New ChromaDB path for BGE-M3 embeddings
CHROMA_DB_PATH = project_root / "data" / "chroma_db_bge_m3"

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

class EmbeddingPipelineBGEM3:
    """Enhanced pipeline with chapter context enrichment for BGE-M3 multi-vector embeddings."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
        }
        
        # Initialize ChromaDB with new path
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        
        # Get or create collection with cosine similarity
        # NOTE: ChromaDB stores dense embeddings; sparse stored in metadata
        self.collection = self.client.get_or_create_collection(
            name=f"hadith_{collection_name}",
            metadata={
                "hnsw:space": "cosine",
                "description": f"{collection_name.title()} hadiths with chapter context - BGE-M3 multi-vector",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dimension": "1024",
                "enrichment": "chapter_titles_prepended",
                "has_sparse": "true",  # Flag that sparse vectors are in metadata
            }
        )
        
        initial_count = self.collection.count()
        logger.info(f"Collection 'hadith_{collection_name}' ready (current: {initial_count} docs)")
        logger.info(f"ChromaDB path: {CHROMA_DB_PATH}")
        
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
    
    def prepare_metadata(self, chunk: Dict[str, Any], sparse_vec: Dict[str, float]) -> Dict[str, Any]:
        """
        Prepare full metadata for ChromaDB storage.
        
        ChromaDB metadata must be str, int, float, or bool.
        Sparse vectors are stored as JSON string in metadata.
        """
        chapter_id = chunk.get("chapter_id")
        chapter_info = self.chapter_map.get(chapter_id, {})
        
        # Serialize sparse vector to JSON string for storage
        sparse_json = json.dumps(sparse_vec) if sparse_vec else "{}"
        
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
            
            # BGE-M3 sparse vector stored as JSON string
            'sparse_vector': sparse_json,
            'sparse_terms_count': len(sparse_vec) if sparse_vec else 0,
        }
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        client: httpx.AsyncClient
    ) -> Optional[Dict[str, Any]]:
        """
        Generate multi-vector embeddings via Modal BGE-M3 API with retry logic.
        
        Returns:
            Dict with 'dense' and 'sparse' lists, or None on failure
        """
        # BGE-M3 does NOT need any prefix - send raw text
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    MODAL_EMBED_URL,
                    json={"texts": texts},
                    timeout=180.0  # Longer timeout for multi-vector
                )
                response.raise_for_status()
                result = response.json()
                
                # Handle error response
                if "error" in result:
                    raise ValueError(f"API error: {result['error']}")
                
                dense = result.get('dense', [])
                sparse = result.get('sparse', [])
                
                if not dense or len(dense) != len(texts):
                    raise ValueError(f"Expected {len(texts)} dense embeddings, got {len(dense) if dense else 0}")
                
                if not sparse or len(sparse) != len(texts):
                    raise ValueError(f"Expected {len(texts)} sparse embeddings, got {len(sparse) if sparse else 0}")
                
                return {'dense': dense, 'sparse': sparse}
                
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
        
        # Generate multi-vector embeddings for ENRICHED texts
        result = await self.generate_embeddings(enriched_texts, client)
        
        if result is None:
            logger.error(f"Batch {batch_num}: Failed to generate embeddings")
            self.stats['failed'] += len(batch)
            return
        
        dense_embeddings = result['dense']
        sparse_embeddings = result['sparse']
        
        # Prepare metadata (includes sparse vectors as JSON)
        metadatas = [
            self.prepare_metadata(chunk, sparse_embeddings[i])
            for i, chunk in enumerate(batch)
        ]
        
        try:
            # Upsert to ChromaDB (store ORIGINAL text as document for display)
            # Dense embeddings go in 'embeddings', sparse stored in metadata
            self.collection.upsert(
                ids=ids,
                embeddings=dense_embeddings,
                documents=original_texts,  # Store original for retrieval display
                metadatas=metadatas,
            )
            
            self.stats['processed'] += len(batch)
            avg_sparse_terms = sum(len(s) for s in sparse_embeddings) / len(sparse_embeddings)
            logger.info(
                f"Batch {batch_num}: {len(batch)} chunks | "
                f"Avg sparse terms: {avg_sparse_terms:.0f} | "
                f"Total: {self.stats['processed']}/{self.stats['total_chunks']}"
            )
            
        except Exception as e:
            logger.error(f"Batch {batch_num}: Storage failed - {e}")
            self.stats['failed'] += len(batch)
    
    async def run(self, chunk_file: Path):
        """Run the embedding pipeline."""
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info(f"BGE-M3 EMBEDDING PIPELINE: {self.collection_name.upper()}")
        logger.info("=" * 70)
        logger.info(f"Model: BAAI/bge-m3 (Multi-Vector: Dense 1024-dim + Sparse)")
        logger.info(f"Chunk file: {chunk_file}")
        logger.info(f"ChromaDB: {CHROMA_DB_PATH}")
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
    logger.info("HADITH EMBEDDING - BGE-M3 Multi-Vector")
    logger.info("Dense (1024-dim) + Sparse (learned BM25-like)")
    logger.info("=" * 70)
    
    all_stats = []
    
    for collection_name, chunk_file in CHUNK_FILES.items():
        if not chunk_file.exists():
            logger.warning(f"Chunk file not found: {chunk_file}. Skipping {collection_name}.")
            continue
        
        pipeline = EmbeddingPipelineBGEM3(collection_name)
        
        try:
            stats = await pipeline.run(chunk_file)
            all_stats.append({
                'collection': collection_name,
                **stats
            })
        except KeyboardInterrupt:
            logger.info("\nPipeline interrupted by user")
            logger.info("Progress saved - you can resume later")
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
        logger.info(f"\nBGE-M3 embeddings stored at: {CHROMA_DB_PATH}")
        logger.info("Each document has:")
        logger.info("  - Dense vector (1024-dim) in ChromaDB embeddings")
        logger.info("  - Sparse vector (JSON) in metadata['sparse_vector']")
        logger.info("\nReady for hybrid search without separate BM25!")
        logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
