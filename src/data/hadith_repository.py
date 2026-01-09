"""
Hadith Repository - Data Access Layer

This module provides a clean abstraction over ChromaDB for hadith data operations.
Follows the Repository Pattern to separate data access concerns from business logic.

Responsibilities:
- Chunk reassembly (combining split hadiths into complete documents)
- Direct hadith lookups by ID
- Metadata-based queries (longest/shortest hadiths)
- Collection access and management

Production Standards:
- Single Responsibility: Only data access, no business logic
- Dependency Injection: ChromaDB client can be injected for testing
- Caching: LRU cache for frequently accessed hadiths
- Error Handling: Graceful fallbacks with logging

Usage:
    from src.data.hadith_repository import HadithRepository
    
    repo = HadithRepository()
    reassembled = repo.reassemble_chunked_hadiths(documents)
    longest = repo.get_longest_hadith(collection="bukhari")
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple
from chromadb.api import ClientAPI
from rapidfuzz import fuzz

from src.tools.retrieval.schemas import Document

logger = logging.getLogger(__name__)


class HadithRepository:
    """
    Repository for hadith data access operations.
    
    Provides a clean abstraction over ChromaDB for:
    - Chunk reassembly
    - Metadata queries
    - Direct lookups
    
    Attributes:
        _client: ChromaDB client instance (injected or singleton)
    """
    
    def __init__(self, chroma_client: Optional[ClientAPI] = None):
        """
        Initialize the repository with optional ChromaDB client injection.
        
        Args:
            chroma_client: Optional ChromaDB client. If not provided,
                          uses the singleton from GlobalClients.
        """
        self._client = chroma_client
        self._collection_cache: Dict[str, Any] = {}
    
    @property
    def client(self) -> ClientAPI:
        """
        Get the ChromaDB client, initializing from singleton if needed.
        
        Returns:
            ChromaDB client instance
        """
        if self._client is None:
            from src.utils.singletons import get_chroma_client
            self._client = get_chroma_client()
        return self._client
    
    def get_collection(self, collection_name: str):
        """
        Get a ChromaDB collection by name, with caching.
        
        Args:
            collection_name: Name of the collection (e.g., "hadith_bukhari")
            
        Returns:
            ChromaDB collection instance
        """
        if collection_name not in self._collection_cache:
            self._collection_cache[collection_name] = self.client.get_collection(collection_name)
        return self._collection_cache[collection_name]
    
    def normalize_collection_name(self, collection: str) -> str:
        """
        Normalize collection name to database format.
        
        Args:
            collection: Collection name (various formats like "Sahih al-Bukhari", "bukhari", etc.)
            
        Returns:
            Normalized database collection name (e.g., "hadith_bukhari")
        """
        coll_lower = collection.lower() if collection else "bukhari"
        if "bukhari" in coll_lower:
            return "hadith_bukhari"
        elif "muslim" in coll_lower:
            return "hadith_muslim"
        return f"hadith_{coll_lower}"
    
    def reassemble_chunked_hadiths(
        self,
        documents: List[Document],
        desired_language: Optional[str] = None,
    ) -> List[Document]:
        """
        Reassemble chunked hadiths into their complete form using BATCH fetching.
        
        When a hadith is too long and was split into multiple chunks during embedding,
        this function fetches all chunks and combines them into the full hadith text.
        
        **Optimized**: Uses a single batch query with $in operator instead of N+1 queries.
        
        Args:
            documents: List of Document objects from search results
            desired_language: Language preference for fetching chunks (arabic/english)
            
        Returns:
            List of Document objects with chunked hadiths reassembled
        """
        if not documents:
            return documents
        
        # Identify documents that need reassembly (total_chunks > 1)
        reassembly_needed = [
            doc for doc in documents 
            if doc.total_chunks and doc.total_chunks > 1
        ]
        
        if not reassembly_needed:
            return documents
        
        logger.info(f"Reassembling {len(reassembly_needed)} chunked hadiths (batch mode)")
        
        try:
            # Group documents by collection for batch fetching
            docs_by_collection: Dict[str, List[Document]] = {}
            for doc in reassembly_needed:
                collection_db_name = self.normalize_collection_name(doc.collection)
                if collection_db_name not in docs_by_collection:
                    docs_by_collection[collection_db_name] = []
                docs_by_collection[collection_db_name].append(doc)
            
            # Batch fetch all chunks for all hadiths at once (per collection)
            all_chunks_map: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}
            
            for collection_db_name, docs in docs_by_collection.items():
                # Collect unique hadith IDs for this collection
                hadith_ids = list(set(doc.hadith_id for doc in docs if doc.hadith_id is not None))
                
                if not hadith_ids:
                    continue
                
                # BATCH FETCH: Single query for all hadith IDs using $in operator
                chunks_data = self.get_chunks_batch(
                    collection_name=collection_db_name,
                    hadith_ids=hadith_ids,
                    language=desired_language,
                )
                
                # Index chunks by (hadith_id, language, collection)
                for chunk in chunks_data:
                    key = (
                        chunk['metadata'].get('hadith_id'),
                        chunk['metadata'].get('language', 'arabic'),
                        chunk['metadata'].get('collection', ''),
                    )
                    if key not in all_chunks_map:
                        all_chunks_map[key] = []
                    all_chunks_map[key].append(chunk)
            
            # Sort chunks within each hadith by chunk_index
            for key in all_chunks_map:
                all_chunks_map[key].sort(key=lambda x: x['metadata'].get('chunk_index', 0))
            
            # Now reassemble documents using the pre-fetched chunks
            reassembled_docs = []
            reassembled_hadith_ids = set()
            
            for doc in documents:
                hadith_key = (doc.hadith_id, doc.language, doc.collection)
                
                # Skip if already reassembled this hadith
                if hadith_key in reassembled_hadith_ids:
                    continue
                
                # If not chunked, keep as-is
                if not doc.total_chunks or doc.total_chunks <= 1:
                    reassembled_docs.append(doc)
                    reassembled_hadith_ids.add(hadith_key)
                    continue
                
                # Get pre-fetched chunks for this hadith
                chunks = all_chunks_map.get(hadith_key, [])
                
                if not chunks:
                    # Fallback: keep original chunk
                    reassembled_docs.append(doc)
                    reassembled_hadith_ids.add(hadith_key)
                    continue
                
                # Combine all chunk texts
                combined_text = "\n".join([chunk['text'] for chunk in chunks])
                
                # Create new Document with combined text
                reassembled_doc = Document(
                    chunk_id=doc.chunk_id,
                    text=combined_text,
                    score=doc.score,
                    search_type=doc.search_type,
                    language=doc.language,
                    collection=doc.collection,
                    book_id=doc.book_id,
                    chapter_id=doc.chapter_id,
                    hadith_id=doc.hadith_id,
                    narrator=doc.narrator,
                    parent_hadith_id=doc.parent_hadith_id,
                    book_number=doc.book_number,
                    chapter_number=doc.chapter_number,
                    hadith_id_in_book=doc.hadith_id_in_book,
                    chunk_index=0,
                    total_chunks=doc.total_chunks,
                    is_chunked=False,
                )
                reassembled_docs.append(reassembled_doc)
                reassembled_hadith_ids.add(hadith_key)
                
                logger.debug(
                    f"Reassembled Hadith #{doc.hadith_id}: "
                    f"{len(chunks)} chunks -> {len(combined_text)} chars"
                )
            
            logger.info(f"Reassembly complete: {len(documents)} -> {len(reassembled_docs)} documents")
            return reassembled_docs
            
        except Exception as e:
            logger.error(f"Chunk reassembly failed: {e}")
            return documents  # Return original on failure
    
    def get_chunks_batch(
        self,
        collection_name: str,
        hadith_ids: List[int],
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch fetch all chunks for multiple hadith IDs in a single query.
        
        Uses ChromaDB's $in operator for efficient batch retrieval.
        
        Args:
            collection_name: Database collection name (e.g., "hadith_bukhari")
            hadith_ids: List of hadith IDs to fetch chunks for
            language: Optional language filter (arabic/english)
            
        Returns:
            List of chunk dictionaries with 'id', 'metadata', 'text' keys
        """
        if not hadith_ids:
            return []
        
        try:
            collection = self.get_collection(collection_name)
            
            # Build where clause with $in for batch fetching
            if language:
                where_clause = {
                    "$and": [
                        {"hadith_id": {"$in": hadith_ids}},
                        {"language": {"$eq": language}}
                    ]
                }
            else:
                where_clause = {"hadith_id": {"$in": hadith_ids}}
            
            result = collection.get(
                where=where_clause,
                include=['metadatas', 'documents']
            )
            
            if not result['ids']:
                return []
            
            # Convert to list of chunk dictionaries
            chunks = []
            for i, (chunk_id, metadata, text) in enumerate(zip(
                result['ids'],
                result['metadatas'],
                result['documents']
            )):
                chunks.append({
                    'id': chunk_id,
                    'metadata': metadata,
                    'text': text,
                })
            
            logger.debug(f"Batch fetched {len(chunks)} chunks for {len(hadith_ids)} hadiths")
            return chunks
            
        except Exception as e:
            logger.error(f"Batch chunk fetch failed: {e}")
            return []
    
    def get_chunks_by_hadith_id(
        self,
        collection_name: str,
        hadith_id: int,
        language: str = "arabic",
    ) -> List[Dict[str, Any]]:
        """
        Fetch all chunks for a specific hadith, sorted by chunk_index.
        
        Args:
            collection_name: Database collection name (e.g., "hadith_bukhari")
            hadith_id: The hadith ID to fetch chunks for
            language: Language filter (arabic/english)
            
        Returns:
            List of chunk dictionaries with 'id', 'metadata', 'text' keys,
            sorted by chunk_index
        """
        try:
            collection = self.get_collection(collection_name)
            
            all_chunks = collection.get(
                where={
                    "$and": [
                        {"hadith_id": {"$eq": hadith_id}},
                        {"language": {"$eq": language}}
                    ]
                },
                include=['metadatas', 'documents']
            )
            
            if not all_chunks['ids']:
                return []
            
            # Sort chunks by chunk_index
            chunk_data = list(zip(
                all_chunks['ids'],
                all_chunks['metadatas'],
                all_chunks['documents']
            ))
            chunk_data.sort(key=lambda x: x[1].get('chunk_index', 0))
            
            return [
                {'id': cid, 'metadata': meta, 'text': text}
                for cid, meta, text in chunk_data
            ]
            
        except Exception as e:
            logger.error(f"Failed to get chunks for hadith {hadith_id}: {e}")
            return []
    
    def get_longest_hadith(
        self,
        collection: str = "bukhari",
        language: Optional[str] = None,
        narrator: Optional[str] = None,
        chapter_id: Optional[int] = None,
    ) -> Optional[Document]:
        """
        Find the longest hadith in a collection by total_chunks and text length.
        
        Args:
            collection: Collection name (bukhari/muslim)
            language: Optional language filter
            narrator: Optional narrator filter (partial match)
            chapter_id: Optional chapter ID filter
            
        Returns:
            Document with the longest hadith, or None if not found
        """
        return self._get_hadith_by_length(
            collection=collection,
            language=language,
            find_longest=True,
            narrator=narrator,
            chapter_id=chapter_id,
        )
    
    def get_shortest_hadith(
        self,
        collection: str = "bukhari",
        language: Optional[str] = None,
        narrator: Optional[str] = None,
        chapter_id: Optional[int] = None,
    ) -> Optional[Document]:
        """
        Find the shortest hadith in a collection by text length.
        
        Args:
            collection: Collection name (bukhari/muslim)
            language: Optional language filter
            narrator: Optional narrator filter (partial match)
            chapter_id: Optional chapter ID filter
            
        Returns:
            Document with the shortest hadith, or None if not found
        """
        return self._get_hadith_by_length(
            collection=collection,
            language=language,
            find_longest=False,
            narrator=narrator,
            chapter_id=chapter_id,
        )
    
    def get_last_hadith(
        self,
        collection: str = "bukhari",
        language: Optional[str] = None,
    ) -> Optional[Document]:
        """
        Find the last hadith in a collection by maximum hadith_id.
        
        Args:
            collection: Collection name (bukhari/muslim)
            language: Optional language filter
            
        Returns:
            Document with the last hadith, or None if not found
        """
        return self._get_hadith_by_position(
            collection=collection,
            language=language,
            find_last=True,
        )
    
    def get_first_hadith(
        self,
        collection: str = "bukhari",
        language: Optional[str] = None,
    ) -> Optional[Document]:
        """
        Find the first hadith in a collection by minimum hadith_id.
        
        Args:
            collection: Collection name (bukhari/muslim)
            language: Optional language filter
            
        Returns:
            Document with the first hadith, or None if not found
        """
        return self._get_hadith_by_position(
            collection=collection,
            language=language,
            find_last=False,
        )
    
    def _get_hadith_by_position(
        self,
        collection: str,
        language: Optional[str],
        find_last: bool,
    ) -> Optional[Document]:
        """
        Internal method to find hadith by position (first or last).
        
        Args:
            collection: Collection name
            language: Optional language filter
            find_last: True for last (max ID), False for first (min ID)
            
        Returns:
            Document or None
        """
        collection_db_name = self.normalize_collection_name(collection)
        
        try:
            db_collection = self.get_collection(collection_db_name)
            
            # Fetch metadatas only to find min/max ID (more efficient)
            result = db_collection.get(include=['metadatas'])
            
            if not result['ids']:
                return None
            
            # Find the entry with min or max hadith_id
            target_hadith_id = None
            target_language = language or "arabic"  # Default to Arabic
            
            for meta in result['metadatas']:
                # Filter by language if needed
                doc_lang = meta.get('language', 'arabic')
                if language and doc_lang != language:
                    continue
                
                hid = meta.get('hadith_id')
                if hid is None or not isinstance(hid, int):
                    continue
                
                if target_hadith_id is None:
                    target_hadith_id = hid
                    target_language = doc_lang
                elif find_last and hid > target_hadith_id:
                    target_hadith_id = hid
                    target_language = doc_lang
                elif not find_last and hid < target_hadith_id:
                    target_hadith_id = hid
                    target_language = doc_lang
            
            if target_hadith_id is None:
                return None
            
            logger.info(f"Found {'last' if find_last else 'first'} hadith: ID={target_hadith_id} in {collection}")
            
            # Fetch and reassemble the hadith chunks
            chunks = self.get_chunks_by_hadith_id(collection_db_name, target_hadith_id, target_language)
            
            if not chunks:
                return None
            
            # Sort chunks by chunk_index
            chunks.sort(key=lambda c: c['metadata'].get('chunk_index', 0))
            
            # Combine all chunk texts
            combined_text = "\n".join([chunk['text'] for chunk in chunks])
            first_meta = chunks[0]['metadata']
            
            return Document(
                chunk_id=chunks[0]['id'],
                text=combined_text,
                score=1.0,
                search_type="metadata_lookup",
                language=first_meta.get('language', 'arabic'),
                collection=first_meta.get('collection', collection),
                hadith_id=first_meta.get('hadith_id'),
                chapter_id=first_meta.get('chapter_id'),
                narrator=first_meta.get('narrator'),
                hadith_id_in_book=first_meta.get('hadith_id_in_book'),
                total_chunks=first_meta.get('total_chunks', 1),
                is_chunked=first_meta.get('is_chunked', False),
            )
            
        except Exception as e:
            logger.error(f"Failed to get {'last' if find_last else 'first'} hadith from {collection}: {e}")
            return None
    
    def _get_hadith_by_length(
        self,
        collection: str,
        language: Optional[str],
        find_longest: bool,
        narrator: Optional[str] = None,
        chapter_id: Optional[int] = None,
    ) -> Optional[Document]:
        """
        Internal method to find hadith by length criteria.
        
        Args:
            collection: Collection name
            language: Optional language filter
            find_longest: True for longest, False for shortest
            narrator: Optional narrator filter (partial match)
            chapter_id: Optional chapter ID filter
            
        Returns:
            Document or None
        """
        collection_db_name = self.normalize_collection_name(collection)
        
        try:
            db_collection = self.get_collection(collection_db_name)
            
            # Get all documents with metadata
            all_docs = db_collection.get(include=['metadatas', 'documents'])
            
            if not all_docs['ids']:
                return None
            
            # Build hadith -> chunks mapping
            hadith_chunks: Dict[int, Dict[str, Any]] = {}
            
            # Keywords that indicate a "reference" or "isnad-only" hadith (noise)
            # e.g., "And narrated similarly...", "With this chain..."
            NOISE_KEYWORDS = [
                "بهذا الإسناد", "مثله", "نحوه", "وحدثناه", "وحدثنا", 
                "similar to", "like it", "same chain"
            ]
            
            for doc_id, meta, text in zip(
                all_docs['ids'],
                all_docs['metadatas'],
                all_docs['documents']
            ):
                hadith_id = meta.get('hadith_id')
                total_chunks = meta.get('total_chunks', 1)
                lang = meta.get('language', 'arabic')
                doc_narrator = meta.get('narrator', '')
                doc_chapter_id = meta.get('chapter_id')
                
                # 1. Filter by language if specified
                if language:
                    if language == 'arabic' and lang != 'arabic':
                        continue
                    if language == 'english' and lang != 'english':
                        continue
                
                # 2. Filter by Narrator (Fuzzy Match)
                if narrator:
                    doc_narrator_lower = doc_narrator.lower() if doc_narrator else ''
                    narrator_lower = narrator.lower()
                    
                    # Fast path: Exact substring match
                    is_match = narrator_lower in doc_narrator_lower
                    
                    # Slow path: Fuzzy match if exact failed
                    if not is_match and doc_narrator:
                        # Check similarity using rapidfuzz (returns 0-100)
                        similarity = fuzz.ratio(narrator_lower, doc_narrator_lower)
                        if similarity > 80:  # Allow small differences like 'h' at the end
                            is_match = True
                    
                    if not is_match:
                        continue
                
                # 3. Filter by Chapter ID (if specified)
                if chapter_id is not None and doc_chapter_id != chapter_id:
                    continue
                
                # 4. Filter out "Noise" when looking for shortest hadith
                # If text is very short (< 100 chars) AND contains reference keywords, skip it
                if not find_longest and text and len(text) < 100:
                    if any(kw in text for kw in NOISE_KEYWORDS):
                        continue
                
                if hadith_id not in hadith_chunks:
                    hadith_chunks[hadith_id] = {
                        'total_chunks': total_chunks,
                        'doc_ids': [doc_id],
                        'metadatas': [meta],
                        'texts': [text],
                        'total_text_length': len(text) if text else 0,
                    }
                else:
                    hadith_chunks[hadith_id]['doc_ids'].append(doc_id)
                    hadith_chunks[hadith_id]['metadatas'].append(meta)
                    hadith_chunks[hadith_id]['texts'].append(text)
                    hadith_chunks[hadith_id]['total_text_length'] += len(text) if text else 0
                    if total_chunks > hadith_chunks[hadith_id]['total_chunks']:
                        hadith_chunks[hadith_id]['total_chunks'] = total_chunks
            
            if not hadith_chunks:
                return None
            
            # Sort hadiths
            if find_longest:
                sorted_hadiths = sorted(
                    hadith_chunks.items(),
                    key=lambda x: (x[1]['total_chunks'], x[1]['total_text_length']),
                    reverse=True
                )
            else:
                sorted_hadiths = sorted(
                    hadith_chunks.items(),
                    key=lambda x: x[1]['total_text_length'],
                    reverse=False
                )
            
            if not sorted_hadiths:
                return None
            
            # Get top result and reassemble
            top_hadith_id, top_data = sorted_hadiths[0]
            
            # Fetch all chunks for complete reassembly
            all_hadith_chunks = self.get_chunks_by_hadith_id(
                collection_name=collection_db_name,
                hadith_id=top_hadith_id,
                language=language or "arabic",
            )
            
            if not all_hadith_chunks:
                # Fallback to original data
                combined_text = "\n".join(top_data['texts'])
                first_meta = top_data['metadatas'][0]
                first_doc_id = top_data['doc_ids'][0]
            else:
                combined_text = "\n".join([c['text'] for c in all_hadith_chunks])
                first_meta = all_hadith_chunks[0]['metadata']
                first_doc_id = all_hadith_chunks[0]['id']
            
            return Document(
                chunk_id=first_doc_id,
                text=combined_text,
                score=1.0,
                search_type="metadata_query",
                language=first_meta.get('language', 'arabic'),
                collection=first_meta.get('collection', ''),
                book_id=first_meta.get('book_id'),
                chapter_id=first_meta.get('chapter_id'),
                hadith_id=first_meta.get('hadith_id'),
                narrator=first_meta.get('narrator'),
                parent_hadith_id=first_meta.get('parent_hadith_id'),
                book_number=first_meta.get('book_number'),
                chapter_number=first_meta.get('chapter_number'),
                hadith_id_in_book=first_meta.get('hadith_id_in_book'),
                chunk_index=0,
                total_chunks=first_meta.get('total_chunks', 1),
                is_chunked=first_meta.get('is_chunked', False),
            )
            
        except Exception as e:
            logger.error(f"Failed to get hadith by length from {collection}: {e}")
            return None
    
    def query_collection(
        self,
        collection_name: str,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a direct query on a collection.
        
        Args:
            collection_name: Database collection name
            where: ChromaDB where clause filter
            include: Fields to include ('metadatas', 'documents', 'embeddings')
            limit: Maximum results to return
            
        Returns:
            ChromaDB query result dictionary
        """
        collection = self.get_collection(collection_name)
        
        kwargs = {}
        if where:
            kwargs['where'] = where
        if include:
            kwargs['include'] = include
        if limit:
            kwargs['limit'] = limit
        
        return collection.get(**kwargs) if kwargs else collection.get(include=include or ['metadatas', 'documents'])


# ============================================================================
# Module-level convenience functions (backward compatibility)
# ============================================================================

# Singleton repository instance
_default_repository: Optional[HadithRepository] = None


def get_hadith_repository() -> HadithRepository:
    """
    Get the default singleton HadithRepository instance.
    
    Returns:
        HadithRepository: Singleton instance
    """
    global _default_repository
    if _default_repository is None:
        _default_repository = HadithRepository()
    return _default_repository


def reassemble_chunked_hadiths(
    documents: List[Document],
    desired_language: Optional[str] = None,
) -> List[Document]:
    """
    Convenience function for chunk reassembly using default repository.
    
    This provides backward compatibility with the original function signature.
    
    Args:
        documents: List of Document objects from search results
        desired_language: Language preference for fetching chunks
        
    Returns:
        List of Document objects with chunked hadiths reassembled
    """
    return get_hadith_repository().reassemble_chunked_hadiths(documents, desired_language)
