"""
Hadith Embedding Script - Phase 2: Embedding and Storage

This script reads chunked hadiths from JSONL files, generates embeddings using
the Modal API, and stores them in ChromaDB with complete metadata.

Features:
- Batch processing for efficient API calls
- Retry logic for failed requests
- Idempotent operations (can re-run safely)
- Progress tracking and statistics
- Separate processing for Bukhari and Muslim collections

Input: ./data/chunks/bukhari_chunks.jsonl, ./data/chunks/muslim_chunks.jsonl
Output: ChromaDB at ./data/chroma_db

Usage:
    python src/data/embed_chunks.py
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

from src.config.settings import MAX_RETRIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
# NOTE: Update this URL after deploying modal_embedding_model.py
# Deploy with: modal deploy zOther/modal_embedding_model.py
# The URL will be: https://[your-username]--multilingual-e5-embeddings-embed.modal.run
MODAL_EMBED_URL = "https://sazaitet110--multilingual-e5-embeddings-embed.modal.run"

CHROMA_DB_PATH = project_root / "data" / "chroma_db"
CHUNK_FILES = {
    # 'bukhari': project_root / "data" / "chunks" / "bukhari_chunks.jsonl",  # COMPLETED
    'muslim': project_root / "data" / "chunks" / "muslim_chunks.jsonl"
}
BATCH_SIZE = 50  # Number of chunks to embed per API call
RETRY_DELAY = 2


class EmbeddingPipeline:
    """Pipeline for generating embeddings and storing in ChromaDB."""
    
    def __init__(self, collection_name: str):
        self.modal_embed_url = MODAL_EMBED_URL
        self.chroma_db_path = CHROMA_DB_PATH
        self.batch_size = BATCH_SIZE
        self.collection_name = collection_name
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'processed': 0,
            'failed': 0
        }
        
        # Initialize ChromaDB
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        logger.info(f"ChromaDB initialized at {CHROMA_DB_PATH}")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=f"hadith_{collection_name}",
            metadata={
                "description": f"{collection_name.title()} hadith collection with Arabic and English chunks",
                "source": "sahih_hadiths",
                "embedding_model": "qwen2.5-14b",
                "chunking": "langchain_recursive",
                "max_chunk_size": 800
            }
        )
        logger.info(f"Collection 'hadith_{collection_name}' ready (current size: {self.collection.count()})")
    
    async def generate_embeddings(self, texts: List[str], client: httpx.AsyncClient) -> Optional[List[List[float]]]:
        """Generate embeddings via Modal API with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    self.modal_embed_url,
                    json={"texts": texts},
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Handle different possible response formats
                if isinstance(result, dict):
                    embeddings = result.get('embeddings', result.get('data', []))
                else:
                    embeddings = result
                
                if not embeddings:
                    raise ValueError("Empty embeddings response")
                
                if len(embeddings) != len(texts):
                    logger.warning(f"Embedding count mismatch: {len(texts)} texts, {len(embeddings)} embeddings")
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All embedding attempts failed for batch")
                    return None
        
        return None
    
    def store_batch(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """Store a batch of chunks with embeddings in ChromaDB."""
        if len(chunks) != len(embeddings):
            logger.error(f"Batch size mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
            return 0
        
        try:
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Use chunk_id directly from the chunk
                ids.append(chunk['chunk_id'])
                documents.append(chunk['text'])
                
                # Prepare metadata for ChromaDB (exclude chunk_id and text)
                metadata = {
                    'collection': chunk.get('collection', ''),
                    'language': chunk.get('language', ''),
                    'book_id': int(chunk.get('book_id', 0)),
                    'chapter_id': int(chunk.get('chapter_id', 0)),
                    'hadith_id': int(chunk.get('hadith_id', 0)),
                    'hadith_id_in_book': int(chunk.get('hadith_id_in_book', 0)),
                    'is_chunked': bool(chunk.get('is_chunked', False)),
                    'chunk_index': int(chunk.get('chunk_index', 0)),
                    'total_chunks': int(chunk.get('total_chunks', 1)),
                    'chunk_size': int(chunk.get('chunk_size', 0)),
                    'parent_hadith_id': chunk.get('parent_hadith_id', ''),
                    'narrator': chunk.get('narrator', ''),
                    'book_number': chunk.get('book_number', ''),
                    'chapter_number': chunk.get('chapter_number', '')
                }
                metadatas.append(metadata)
            
            # Upsert to ChromaDB (idempotent)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error storing batch: {e}")
            return 0
    
    async def process_chunks_file(self, chunk_file: Path):
        """Process chunks from JSONL file in batches."""
        logger.info(f"Reading chunks from: {chunk_file}")
        
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
                    continue
        
        logger.info(f"Loaded {len(chunks)} chunks from file")
        
        # Process in batches
        async with httpx.AsyncClient() as client:
            batch_num = 0
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                batch_num += 1
                await self._process_batch(batch, batch_num, client)
    
    async def _process_batch(self, batch: List[Dict[str, Any]], batch_num: int, client: httpx.AsyncClient):
        """Process a single batch."""
        logger.info(f"Batch {batch_num}: Processing {len(batch)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in batch]
        
        # Generate embeddings
        embeddings = await self.generate_embeddings(texts, client)
        
        if embeddings is None:
            logger.error(f"Batch {batch_num}: Failed to generate embeddings")
            self.stats['failed'] += len(batch)
            return
        
        # Store in ChromaDB
        stored = self.store_batch(batch, embeddings)
        
        if stored > 0:
            self.stats['processed'] += stored
            logger.info(f"Batch {batch_num}: Stored {stored} chunks | Total: {self.stats['processed']}/{self.stats['total_chunks']}")
        else:
            self.stats['failed'] += len(batch)
            logger.error(f"Batch {batch_num}: Failed to store")
    
    async def run(self, chunk_file: Path):
        """Run the embedding pipeline."""
        start_time = time.time()
        
        logger.info("="*70)
        logger.info(f"Processing: {self.collection_name.title()}")
        logger.info("="*70)
        logger.info(f"Input: {chunk_file}")
        logger.info(f"Batch size: {BATCH_SIZE}")
        logger.info(f"Modal API: {self.modal_embed_url}")
        logger.info("="*70)
        
        try:
            await self.process_chunks_file(chunk_file)
            
            duration = time.time() - start_time
            final_count = self.collection.count()
            
            logger.info("\n" + "="*70)
            logger.info(f"{self.collection_name.title()} COMPLETE!")
            logger.info("="*70)
            logger.info(f"Total chunks read: {self.stats['total_chunks']}")
            logger.info(f"Successfully embedded: {self.stats['processed']}")
            logger.info(f"Failed: {self.stats['failed']}")
            logger.info(f"ChromaDB total: {final_count} documents")
            logger.info(f"Duration: {duration:.2f} seconds")
            if self.stats['total_chunks'] > 0:
                logger.info(f"Average: {duration/self.stats['total_chunks']:.3f} seconds per chunk")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


async def main():
    """Main entry point."""
    logger.info("\n" + "="*70)
    logger.info("HADITH EMBEDDING - Phase 2")
    logger.info("Generating Embeddings and Storing in ChromaDB")
    logger.info("="*70)
    
    all_stats = []
    
    # Process each collection
    for collection_name, chunk_file in CHUNK_FILES.items():
        if not chunk_file.exists():
            logger.warning(f"Chunks file not found: {chunk_file}. Skipping {collection_name}.")
            continue
        
        pipeline = EmbeddingPipeline(collection_name)
        
        try:
            await pipeline.run(chunk_file)
            all_stats.append({
                'collection': collection_name,
                **pipeline.stats
            })
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.error(f"Pipeline failed for {collection_name}: {e}")
            continue
    
    # Overall summary
    if all_stats:
        logger.info("\n" + "="*70)
        logger.info("OVERALL SUMMARY")
        logger.info("="*70)
        
        total_chunks = sum(s['total_chunks'] for s in all_stats)
        total_processed = sum(s['processed'] for s in all_stats)
        total_failed = sum(s['failed'] for s in all_stats)
        
        logger.info(f"Total Chunks: {total_chunks}")
        logger.info(f"Successfully Processed: {total_processed}")
        logger.info(f"Failed: {total_failed}")
        logger.info("\nPhase 2 Complete! Embeddings stored in ChromaDB.")
        logger.info("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
