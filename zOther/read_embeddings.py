"""
Script to read and explore embeddings from ChromaDB.
"""
import sys
from pathlib import Path
import chromadb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

CHROMA_DB_PATH = project_root / "data" / "chroma_db"

def explore_collections():
    """List all collections and their statistics."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    collections = client.list_collections()
    
    print("\n" + "="*70)
    print("CHROMADB COLLECTIONS")
    print("="*70)
    
    for collection in collections:
        print(f"\nCollection: {collection.name}")
        print(f"  Total embeddings: {collection.count()}")
        print(f"  Metadata: {collection.metadata}")
    
    return collections


def sample_embeddings(collection_name: str = "hadith_bukhari", n: int = 5):
    """Get sample embeddings from a collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"\nError: Collection '{collection_name}' not found!")
        print(f"Available collections: {[c.name for c in client.list_collections()]}")
        return
    
    print("\n" + "="*70)
    print(f"SAMPLE EMBEDDINGS FROM: {collection_name}")
    print("="*70)
    
    # Get first n items
    results = collection.get(
        limit=n,
        include=["embeddings", "metadatas", "documents"]
    )
    
    print(f"\nTotal items retrieved: {len(results['ids'])}")
    
    for i, (id_, doc, metadata, embedding) in enumerate(zip(
        results['ids'], 
        results['documents'], 
        results['metadatas'], 
        results['embeddings']
    ), 1):
        print(f"\n--- Item {i} ---")
        print(f"ID: {id_}")
        print(f"Document (first 100 chars): {doc[:100]}...")
        print(f"\nAll metadata keys: {list(metadata.keys())}")
        print(f"\nMetadata:")
        for key, value in metadata.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
        print(f"\nEmbedding dimension: {len(embedding)}")
        print(f"Embedding (first 10 values): {embedding[:10]}")


def search_similar(query_text: str, collection_name: str = "hadith_bukhari", n_results: int = 3):
    """Search for similar hadiths using text query."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"\nError: Collection '{collection_name}' not found!")
        return
    
    print("\n" + "="*70)
    print(f"SEMANTIC SEARCH IN: {collection_name}")
    print("="*70)
    print(f"Query: '{query_text}'")
    print(f"Requesting {n_results} most similar results...")
    
    # Note: This requires ChromaDB to have an embedding function configured
    # For now, we'll just show how to query by ID
    print("\nNote: Direct text search requires embedding the query first.")
    print("Use the Modal API to embed your query, then use query_embeddings().")


def query_by_embedding(embedding: list, collection_name: str = "hadith_bukhari", n_results: int = 3):
    """Query using a pre-computed embedding vector."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"\nError: Collection '{collection_name}' not found!")
        return
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    print("\n" + "="*70)
    print(f"QUERY RESULTS FROM: {collection_name}")
    print("="*70)
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\n--- Result {i} (Distance: {distance:.4f}) ---")
        print(f"Document: {doc[:200]}...")
        print(f"Language: {metadata.get('language', 'N/A')}")
        print(f"Book: {metadata.get('book_name', 'N/A')}")
        print(f"Chapter: {metadata.get('chapter_name', 'N/A')}")


def get_stats():
    """Get detailed statistics about the embeddings."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    print("\n" + "="*70)
    print("CHROMADB STATISTICS")
    print("="*70)
    
    collections = client.list_collections()
    
    total_embeddings = 0
    for collection in collections:
        count = collection.count()
        total_embeddings += count
        
        print(f"\n{collection.name}:")
        print(f"  Total embeddings: {count}")
        
        # Sample some items to check languages
        sample = collection.get(limit=100, include=["metadatas"])
        if sample['metadatas']:
            languages = {}
            for meta in sample['metadatas']:
                lang = meta.get('language', 'unknown')
                languages[lang] = languages.get(lang, 0) + 1
            
            print(f"  Language distribution (sample of 100):")
            for lang, count in languages.items():
                print(f"    {lang}: {count}")
    
    print(f"\nTotal embeddings across all collections: {total_embeddings}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore ChromaDB embeddings")
    parser.add_argument("--action", choices=["list", "sample", "stats"], 
                       default="stats", help="Action to perform")
    parser.add_argument("--collection", default="hadith_bukhari", 
                       help="Collection name")
    parser.add_argument("--limit", type=int, default=5, 
                       help="Number of samples to show")
    
    args = parser.parse_args()
    
    if args.action == "list":
        explore_collections()
    elif args.action == "sample":
        sample_embeddings(args.collection, args.limit)
    elif args.action == "stats":
        get_stats()
