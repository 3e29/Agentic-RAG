import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration - adjust these paths to match your local environment
RAW_DATA_PATHS = {
    'Bukhari': 'data/raw/bukhari.json',
    'Muslim': 'data/raw/muslim.json'
}

# Max chunk size used in your project
MAX_CHUNK_SIZE = 800

def load_data():
    all_hadiths = []
    for collection, path in RAW_DATA_PATHS.items():
        if not Path(path).exists():
            print(f"Warning: {path} not found.")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Map chapter IDs to names for Sahih Bukhari
            chapters = {c['id']: c['english'] for c in data.get('chapters', [])}
            
            for h in data.get('hadiths', []):
                # Extract text
                arabic_text = h.get('arabic', '')
                
                # Check if it would be chunked
                is_chunked = len(arabic_text) > MAX_CHUNK_SIZE
                
                all_hadiths.append({
                    'collection': collection,
                    'length': len(arabic_text),
                    'is_chunked': is_chunked,
                    'book_name': chapters.get(h.get('chapterId'), f"Book {h.get('bookId')}")
                })
    return pd.DataFrame(all_hadiths)

df = load_data()

# Set visual style
sns.set_theme(style="whitegrid")

# --- CHART 1: Distribution of Hadith Lengths ---
plt.figure(figsize=(10, 6))
# Using fixed bin width of 100 for "proper steps" in hundreds
sns.histplot(df['length'], binwidth=100, kde=True, color='skyblue')
plt.title('Distribution of Hadith Lengths (Character Count)')
plt.xlabel('Character Length')
plt.ylabel('Frequency')
plt.xticks(range(0, 1100, 100)) # Set ticks every 100 units
plt.xlim(0, 1000) 
plt.savefig('hadith_length_distribution.png')

# --- CHART 2 & 3: Top 10 Books (Separate for Bukhari & Muslim) ---
for coll in ['Bukhari', 'Muslim']:
    plt.figure(figsize=(12, 8))
    top_books = df[df['collection'] == coll]['book_name'].value_counts().head(10)
    sns.barplot(x=top_books.values, y=top_books.index, hue=top_books.index, palette='viridis', legend=False)
    plt.title(f'Top 10 Books in Sahih {coll} by Hadith Count')
    plt.xlabel('Hadith Count')
    plt.ylabel('Book Name')
    plt.tight_layout()
    plt.savefig(f'top_10_books_{coll.lower()}.png')

# --- CHART 4: Visualization of Chunked vs Unchunked Hadiths ---
plt.figure(figsize=(8, 8))
chunk_counts = df['is_chunked'].value_counts()
labels = ['Unchunked (Short)', 'Chunked (Long)']
plt.pie(chunk_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999'])
plt.title(f'Proportion of Hadiths Requiring Chunking (Limit: {MAX_CHUNK_SIZE} chars)')
plt.savefig('hadith_chunking_proportion.pie.png')

plt.show()