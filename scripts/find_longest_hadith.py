"""
Find the longest hadith in Bukhari based on total_chunks metadata.
"""
import chromadb

client = chromadb.PersistentClient(path='./data/chroma_db_bge_m3')
coll = client.get_collection('hadith_bukhari')

# Get all docs with metadata
result = coll.get(include=['metadatas'])
print(f'Total documents: {len(result["ids"])}')

# Find docs with most chunks
chunk_counts = {}
for i, meta in enumerate(result['metadatas']):
    hadith_id = meta.get('hadith_id')
    tc = meta.get('total_chunks', 1)
    if hadith_id not in chunk_counts:
        chunk_counts[hadith_id] = tc

# Sort by chunk count
sorted_hadiths = sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True)

print(f'\nTop 10 hadiths by chunk count (longest):')
for hadith_id, tc in sorted_hadiths[:10]:
    print(f'  Hadith #{hadith_id}: {tc} chunks')

print(f'\n\nUnique hadith IDs: {len(chunk_counts)}')
