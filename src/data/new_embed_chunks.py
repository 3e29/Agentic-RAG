import json
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import httpx
import chromadb

# --- CONFIGURATION ---
# Define paths to BOTH the chunks and the original source files
DATA_DIR = Path("data")
CHUNK_FILES = {
    "bukhari": DATA_DIR / "chunks" / "bukhari_chunks.jsonl",
    "muslim": DATA_DIR / "chunks" / "muslim_chunks.jsonl",
}
# We need these to look up Chapter Names (e.g., ID 24 -> "Book of Zakat")
SOURCE_FILES = {
    "bukhari": DATA_DIR / "sahih_bukhari.json", # Update with your actual filename
    "muslim": DATA_DIR / "sahih_muslim.json",   # Update with your actual filename
}

# Your Modal Endpoint
EMBEDDING_API_URL = "https://sazaitet110--multilingual-e5-embeddings-embed.modal.run" 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_chapter_mapping(source_path: Path) -> Dict[int, Dict[str, str]]:
    """
    Reads the original JSON file to build a map: 
    ChapterID -> {'en': 'Book of Zakat', 'ar': 'كتاب الزكاة'}
    """
    if not source_path.exists():
        logger.warning(f"⚠️ Source file not found: {source_path}. Metadata will be incomplete.")
        return {}
    
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        mapping = {}
        for chapter in data.get("chapters", []):
            c_id = chapter.get("id")
            # Handle different field names in source (arabic/english vs ar/en)
            ar_title = chapter.get("arabic", "")
            en_title = chapter.get("english", "")
            
            mapping[c_id] = {
                "chapter_title_en": en_title,
                "chapter_title_ar": ar_title
            }
        logger.info(f"✅ Loaded {len(mapping)} chapter titles from {source_path.name}")
        return mapping
    except Exception as e:
        logger.error(f"❌ Failed to load source map: {e}")
        return {}

class EmbeddingPipeline:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load the Chapter Map for this collection
        source_key = "bukhari" if "bukhari" in collection_name.lower() else "muslim"
        self.chapter_map = load_chapter_mapping(SOURCE_FILES.get(source_key, Path("invalid")))
        
        self.stats = {'total_chunks': 0, 'processed': 0, 'failed': 0, 'skipped': 0}

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Call Modal Endpoint with retries"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(EMBEDDING_API_URL, json={"text": texts}) # Check if your API expects "text" or "queries"
                    resp.raise_for_status()
                    return resp.json() # Assumes returns list of vectors
                except Exception as e:
                    if attempt == 2: raise e
                    await asyncio.sleep(2 ** attempt)

    async def run(self, input_file: Path):
        logger.info(f"🚀 Starting pipeline for {self.collection_name}...")
        
        batch_size = 32 # Process in batches
        current_batch = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    self.stats['total_chunks'] += 1
                    
                    # --- CRITICAL FIX: DATA ENRICHMENT ---
                    chapter_id = chunk.get("chapter_id")
                    chapter_info = self.chapter_map.get(chapter_id, {})
                    
                    # 1. Enrich Metadata
                    chunk["chapter_title_en"] = chapter_info.get("chapter_title_en", "Unknown")
                    chunk["chapter_title_ar"] = chapter_info.get("chapter_title_ar", "Unknown")
                    
                    # 2. Enrich Text (So Semantic Search finds "Zakat" even if hadith text doesn't say it)
                    # We prepend the Context to the text
                    enriched_text = f"Chapter: {chunk['chapter_title_en']} ({chunk['chapter_title_ar']})\n{chunk['text']}"
                    
                    current_batch.append({
                        "id": chunk["chunk_id"],
                        "text": enriched_text, # Embed the ENRICHED text
                        "metadata": {
                            "hadith_id": chunk["hadith_id"],
                            "chapter_id": chunk["chapter_id"],
                            "chapter_title_en": chunk["chapter_title_en"], # Store for filtering
                            "source": self.collection_name
                        }
                    })

                    if len(current_batch) >= batch_size:
                        await self._process_batch(current_batch)
                        current_batch = []
                        
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    self.stats['failed'] += 1

            if current_batch:
                await self._process_batch(current_batch)

    async def _process_batch(self, batch):
        try:
            texts = [item["text"] for item in batch]
            ids = [item["id"] for item in batch]
            metadatas = [item["metadata"] for item in batch]
            
            embeddings = await self.embed_batch(texts)
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            self.stats['processed'] += len(batch)
            print(f"Processed {self.stats['processed']}...", end="\r")
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            self.stats['failed'] += len(batch)

if __name__ == "__main__":
    # Simple runner
    async def main():
        for name, path in CHUNK_FILES.items():
            if path.exists():
                await EmbeddingPipeline(name).run(path)
    asyncio.run(main())