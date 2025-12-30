"""
Data Ingestion Pipeline for Hadith RAG System

This module implements the complete data ingestion pipeline:
1. Load and parse hadith data from JSON (FR-DIC-32)
2. Preprocess Arabic text (FR-DIC-33)
3. Generate embeddings via Modal API (FR-DIC-34)
4. Store in ChromaDB vector database (FR-DIC-35)

Usage:
    python -m src.data.ingestion

Requirements Implemented:
    - FR-DIC-31: Data Ingestion Pipeline
    - FR-DIC-32: JSON Parsing
    - FR-DIC-33: Text Preprocessing
    - FR-DIC-34: Embedding Generation
    - FR-DIC-35: Vector Storage
    - NFR-05: Data Consistency
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

import httpx
import chromadb
from chromadb.config import Settings

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.arabic_processing import normalize_arabic_text
from src.utils.chunking import chunk_hadith, get_chunk_statistics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Configuration
MODAL_EMBED_URL = "https://sazaitet110--qwen2-5-14b-runner-qwenmodel-embed.modal.run"
CHROMA_DB_PATH = "./data/chroma_db"
RAW_DATA_PATHS = [
    "./data/raw/bukhari.json",
    "./data/raw/muslim.json"
]
COLLECTION_NAME = "hadith_collection"
BATCH_SIZE = 50  # Number of chunks to embed in one batch
MAX_CHUNK_SIZE = 800  # Maximum characters per chunk (≈400 tokens)
CHUNK_OVERLAP = 1  # Number of sentences to overlap between chunks
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class HadithIngestionPipeline:
    """
    Main pipeline for ingesting Hadith data into ChromaDB.
    """
    
    def __init__(
        self,
        modal_embed_url: str = MODAL_EMBED_URL,
        chroma_db_path: str = CHROMA_DB_PATH,
        batch_size: int = BATCH_SIZE,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            modal_embed_url: URL of the Modal embedding endpoint
            chroma_db_path: Path to ChromaDB persistent storage
            batch_size: Number of chunks to embed per batch
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Number of sentences to overlap between chunks
        """
        self.modal_embed_url = modal_embed_url
        self.chroma_db_path = chroma_db_path
        self.batch_size = batch_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Statistics
        self.stats = {
            "total_hadiths": 0,
            "total_chunks": 0,
            "chunked_hadiths": 0,
            "processed": 0,
            "failed": 0,
            "skipped": 0
        }
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        logger.info(f"Initialized ChromaDB client at {chroma_db_path}")
        
        # Initialize HTTP client for Modal
        self.http_client = httpx.Client(timeout=60.0)
        
    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load and parse the Hadith JSON file.
        
        Implements FR-DIC-32: JSON Parsing
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON data as dictionary
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Successfully loaded JSON with {len(data.get('chapters', []))} chapters")
            return data
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
    
    def extract_hadiths(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all hadith records from the JSON structure.
        
        Each hadith is extracted with its complete metadata for citation purposes.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            List of hadith records with metadata
        """
        hadiths = []
        
        # Get book-level metadata
        book_metadata = data.get('metadata', {})
        book_id = data.get('id')
        
        # Create chapter lookup for faster access
        chapters_dict = {
            ch['id']: ch for ch in data.get('chapters', [])
        }
        
        # Extract hadiths (they're in the 'hadiths' field at root level)
        # Based on the JSON structure, hadiths are at root level
        for hadith in data.get('hadiths', []):
            chapter_id = hadith.get('chapterId')
            chapter_info = chapters_dict.get(chapter_id, {})
            
            # Create hadith record with complete metadata
            hadith_record = {
                'id': f"bukhari_{book_id}_{hadith.get('id')}",
                'hadith_id': hadith.get('id'),
                'hadith_id_in_book': hadith.get('idInBook'),
                'book_id': book_id,
                'chapter_id': chapter_id,
                'chapter_arabic': chapter_info.get('arabic', ''),
                'chapter_english': chapter_info.get('english', ''),
                'collection_arabic': book_metadata.get('arabic', {}).get('title', ''),
                'collection_english': book_metadata.get('english', {}).get('title', ''),
                'author_arabic': book_metadata.get('arabic', {}).get('author', ''),
                'author_english': book_metadata.get('english', {}).get('author', ''),
                'text_arabic': hadith.get('arabic', ''),
                'text_english': hadith.get('english', {}).get('text', ''),
                'narrator_english': hadith.get('english', {}).get('narrator', ''),
            }
            
            hadiths.append(hadith_record)
        
        self.stats['total_hadiths'] = len(hadiths)
        logger.info(f"Extracted {len(hadiths)} hadiths from JSON")
        
        return hadiths
    
    def preprocess_and_chunk_hadith(self, hadith: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize and chunk hadith text for BOTH Arabic and English.
        
        Implements FR-DIC-33: Text Preprocessing
        Implements semantic chunking for long hadiths
        Processes both languages to support bilingual retrieval
        
        NOTE: The raw JSON files have already been preprocessed with script.py,
        script2.py, and script3.py to remove diacritics and directional marks.
        This function only normalizes whitespace and applies chunking.
        
        Args:
            hadith: Hadith record from preprocessed JSON
            
        Returns:
            List of chunk dictionaries for both Arabic and English
        """
        all_chunks = []
        
        # Prepare base metadata (shared between Arabic and English)
        base_metadata = {
            'hadith_id': hadith['hadith_id'],
            'hadith_id_in_book': hadith['hadith_id_in_book'],
            'book_id': hadith['book_id'],
            'chapter_id': hadith['chapter_id'],
            'chapter_arabic': normalize_arabic_text(hadith['chapter_arabic']),
            'chapter_english': hadith['chapter_english'],
            'collection_arabic': normalize_arabic_text(hadith['collection_arabic']),
            'collection_english': hadith['collection_english'],
            'author_arabic': hadith['author_arabic'],
            'author_english': hadith['author_english'],
            'narrator_english': hadith['narrator_english']
        }
        
        # Process ARABIC text
        text_arabic_cleaned = normalize_arabic_text(hadith['text_arabic'])
        if text_arabic_cleaned:
            arabic_metadata = {
                **base_metadata,
                'text_english': hadith['text_english']  # Keep English for reference
            }
            arabic_chunks = chunk_hadith(
                text_arabic_cleaned,
                arabic_metadata,
                max_chunk_size=self.max_chunk_size,
                overlap_sentences=self.chunk_overlap,
                language='arabic'
            )
            all_chunks.extend(arabic_chunks)
        
        # Process ENGLISH text
        text_english = hadith['text_english'].strip()
        if text_english:
            english_metadata = {
                **base_metadata,
                'text_arabic': hadith['text_arabic']  # Keep Arabic for reference
            }
            english_chunks = chunk_hadith(
                text_english,
                english_metadata,
                max_chunk_size=self.max_chunk_size,
                overlap_sentences=self.chunk_overlap,
                language='english'
            )
            all_chunks.extend(english_chunks)
        
        # Track statistics (chunked if more than just 1 Arabic + 1 English)
        if len(all_chunks) > 2:
            self.stats['chunked_hadiths'] += 1
        self.stats['total_chunks'] += len(all_chunks)
        
        return all_chunks
    
    def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a batch of texts using Modal API.
        
        Implements FR-DIC-34: Embedding Generation
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors, or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Generating embeddings for {len(texts)} texts (attempt {attempt + 1})")
                
                # Call Modal embedding endpoint
                response = self.http_client.post(
                    self.modal_embed_url,
                    json={"text": texts}
                )
                
                response.raise_for_status()
                
                # Extract embeddings from response
                embeddings = response.json().get('embeddings', [])
                
                if len(embeddings) != len(texts):
                    logger.warning(
                        f"Embedding count mismatch: sent {len(texts)}, received {len(embeddings)}"
                    )
                
                logger.debug(f"Successfully generated {len(embeddings)} embeddings")
                return embeddings
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error during embedding generation: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error during embedding generation: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
        
        return None
    
    def store_in_chromadb(
        self,
        collection,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Store chunks and embeddings in ChromaDB.
        
        Implements FR-DIC-35: Vector Storage
        Ensures NFR-05: Data Consistency
        
        Args:
            collection: ChromaDB collection
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors
            
        Returns:
            Number of successfully stored chunks
        """
        if len(chunks) != len(embeddings):
            logger.error(
                f"Count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
            return 0
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Generate unique ID for chunk (includes language)
                chunk_meta = chunk['metadata']
                parent_id = chunk_meta.get('parent_hadith_id', '')
                language = chunk_meta.get('language', 'arabic')
                
                if chunk_meta.get('is_chunked', False):
                    chunk_id = f"{parent_id}_{language}_chunk_{chunk_meta['chunk_index']}"
                else:
                    chunk_id = f"{parent_id}_{language}"
                
                ids.append(chunk_id)
                documents.append(chunk['text'])
                metadatas.append(chunk_meta)
            
            # Add to collection (upsert to handle re-runs idempotently)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.debug(f"Successfully stored {len(chunks)} chunks in ChromaDB")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error storing chunks in ChromaDB: {e}")
            return 0
    
    def process_batch(
        self,
        collection,
        hadiths: List[Dict[str, Any]]
    ) -> int:
        """
        Process a batch of hadiths: preprocess, chunk, embed, and store.
        
        Args:
            collection: ChromaDB collection
            hadiths: Batch of hadith records
            
        Returns:
            Number of successfully processed chunks
        """
        # Preprocess and chunk all hadiths
        all_chunks = []
        for hadith in hadiths:
            chunks = self.preprocess_and_chunk_hadith(hadith)
            all_chunks.extend(chunks)
        
        logger.debug(f"Created {len(all_chunks)} chunks from {len(hadiths)} hadiths")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in all_chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        if embeddings is None:
            logger.error(f"Failed to generate embeddings for batch")
            self.stats['failed'] += len(all_chunks)
            return 0
        
        # Store in ChromaDB
        stored_count = self.store_in_chromadb(collection, all_chunks, embeddings)
        
        if stored_count > 0:
            self.stats['processed'] += stored_count
        else:
            self.stats['failed'] += len(all_chunks)
        
        return stored_count
    
    def run(self, data_paths: List[str] = None) -> Dict[str, int]:
        """
        Run the complete ingestion pipeline for all collections.
        
        Args:
            data_paths: List of paths to raw JSON data files (default: bukhari and muslim)
            
        Returns:
            Statistics dictionary with processing results
        """
        if data_paths is None:
            data_paths = RAW_DATA_PATHS
        
        logger.info("=" * 80)
        logger.info("Starting Hadith Data Ingestion Pipeline")
        logger.info(f"Collections to process: {len(data_paths)}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Get or create ChromaDB collection (shared across all collections)
            logger.info(f"Initializing ChromaDB collection: {COLLECTION_NAME}")
            collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "description": "Hadith embeddings for RAG system",
                    "sources": "Sahih al-Bukhari, Sahih Muslim",
                    "language": "arabic",
                    "chunking": "semantic",
                    "max_chunk_size": self.max_chunk_size
                }
            )
            
            existing_count = collection.count()
            logger.info(f"Collection currently contains {existing_count} documents")
            
            # Process each collection
            for data_path in data_paths:
                logger.info("\n" + "=" * 80)
                logger.info(f"Processing: {data_path}")
                logger.info("=" * 80)
                
                # Step 1: Load JSON data
                data = self.load_json_data(data_path)
                
                # Step 2: Extract hadiths
                hadiths = self.extract_hadiths(data)
                
                if not hadiths:
                    logger.error(f"No hadiths found in {data_path}")
                    continue
            
                # Step 3: Process in batches
                collection_name = Path(data_path).stem  # bukhari or muslim
                logger.info(f"Processing {len(hadiths)} hadiths from {collection_name} in batches of {self.batch_size}")
            
                for i in range(0, len(hadiths), self.batch_size):
                    batch = hadiths[i:i + self.batch_size]
                    batch_num = (i // self.batch_size) + 1
                    total_batches = (len(hadiths) + self.batch_size - 1) // self.batch_size
                    
                    logger.info(f"[{collection_name}] Batch {batch_num}/{total_batches} ({len(batch)} hadiths)")
                    
                    self.process_batch(collection, batch)
                    
                    # Progress update
                    logger.info(
                        f"Progress: {self.stats['processed']} chunks | "
                        f"{self.stats['total_hadiths']} hadiths | Failed: {self.stats['failed']}"
                    )
            
            # Final statistics
            duration = time.time() - start_time
            final_count = collection.count()
            
            logger.info("\n" + "=" * 80)
            logger.info("Ingestion Complete!")
            logger.info("=" * 80)
            logger.info(f"Total hadiths processed: {self.stats['total_hadiths']}")
            logger.info(f"Total chunks created: {self.stats['total_chunks']}")
            logger.info(f"Hadiths that were chunked: {self.stats['chunked_hadiths']}")
            logger.info(f"Successfully stored: {self.stats['processed']} chunks")
            logger.info(f"Failed: {self.stats['failed']}")
            logger.info(f"Collection total: {final_count} documents")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Average: {duration / self.stats['total_hadiths']:.3f} seconds per hadith")
            logger.info("=" * 80)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Fatal error during ingestion: {e}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            self.http_client.close()
    
    def verify_storage(self) -> bool:
        """
        Verify that data was stored correctly in ChromaDB.
        
        Returns:
            True if verification passes, False otherwise
        """
        try:
            collection = self.client.get_collection(name=COLLECTION_NAME)
            count = collection.count()
            
            logger.info(f"Verification: Collection contains {count} documents")
            
            # Test query
            if count > 0:
                results = collection.query(
                    query_texts=["الإيمان"],
                    n_results=1
                )
                
                if results and results['documents']:
                    logger.info(f"Verification: Sample query successful")
                    logger.info(f"Sample result: {results['documents'][0][:100]}...")
                    return True
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False


def main():
    """Main entry point for the ingestion pipeline."""
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize and run pipeline
    pipeline = HadithIngestionPipeline()
    
    try:
        stats = pipeline.run()
        
        # Verify storage
        if pipeline.verify_storage():
            logger.info("✅ Storage verification passed")
        else:
            logger.warning("⚠️ Storage verification failed")
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            logger.warning(f"Pipeline completed with {stats['failed']} failures")
            sys.exit(1)
        else:
            logger.info("✅ Pipeline completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
