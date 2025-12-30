# GTE Multilingual Embedding Model Deployment Guide

This guide explains how to deploy the **Alibaba-NLP/gte-multilingual-base** embedding model to Modal.com.

## Model Information

| Property                | Value                                                   |
| ----------------------- | ------------------------------------------------------- |
| **Model**               | Alibaba-NLP/gte-multilingual-base                       |
| **Purpose**             | Generate embeddings for Arabic and English hadith texts |
| **Embedding Dimension** | 768                                                     |
| **Max Sequence Length** | 8192 tokens                                             |
| **Parameters**          | 305M                                                    |
| **Languages**           | 70+ including Arabic and English                        |
| **Architecture**        | Encoder-only transformer                                |

## Why GTE over E5?

| Feature                 | GTE Multilingual Base | E5 Large Multilingual |
| ----------------------- | --------------------- | --------------------- |
| **Embedding Dimension** | 768                   | 1024                  |
| **Max Tokens**          | 8192 ✅               | 512                   |
| **Parameters**          | 305M                  | 560M                  |
| **Prefix Required**     | No ✅                 | Yes (passage:/query:) |
| **Memory Usage**        | Lower ✅              | Higher                |

**Key Advantages:**

1. **16x longer context** (8192 vs 512 tokens) - better for long hadiths
2. **No instruction prefixes** - simpler to use
3. **Smaller dimensions** - more efficient storage and retrieval
4. **Strong multilingual performance** - especially for Arabic

## Prerequisites

1. **Modal Account**: Sign up at https://modal.com
2. **Modal CLI**: Install the Modal Python package
   ```bash
   pip install modal
   ```

## Deployment Steps

### 1. Setup Modal Token

First time setup - authenticate with Modal:

```bash
modal token new
```

This will open a browser window to authenticate. Follow the instructions.

### 2. Deploy the GTE Embedding Model

Deploy the model to Modal:

```bash
modal deploy zOther/modal_gte_embedding_model.py
```

This will:

- Create a Modal app called **"gte-multilingual-embeddings"**
- Download the model from Hugging Face (~1.2GB)
- Cache it in a Modal Volume for reuse
- Deploy a web endpoint for API access

**Expected output:**

```
✓ Created objects.
├── 🔨 Created mount /path/to/project
├── 🔨 Created download_model.
├── 🔨 Created EmbeddingModel.
├── 🔨 Created test_embeddings.
└── 🔨 Created web function embed => https://[your-username]--gte-multilingual-embeddings-embed.modal.run
✓ App deployed! 🎉
```

### 3. Copy Your Endpoint URL

After successful deployment, Modal will provide a URL like:

```
https://[your-username]--gte-multilingual-embeddings-embed.modal.run
```

**Copy this URL** - you'll need it for testing and embedding.

### 4. Test the Deployment

Test the model with sample texts:

```bash
modal run zOther/modal_gte_embedding_model.py
```

**Expected output:**

```
Testing GTE embedding generation...

Generated 5 embeddings
Embedding dimension: 768
First embedding (first 10 values): [0.123, -0.456, ...]

Cosine similarity 'Bismillah' Arabic vs English: 0.8XXX
Cosine similarity 'Patience/Gratitude' Arabic vs English: 0.8XXX

Test completed successfully!
```

### 5. Test the Web Endpoint

Update the URL in `test_gte_endpoint.py` and run:

```bash
python zOther/test_gte_endpoint.py
```

## API Usage

### Request Format

**Batch request (recommended for efficiency):**

```python
import httpx

response = httpx.post(
    "https://[your-username]--gte-multilingual-embeddings-embed.modal.run",
    json={"texts": ["text1", "text2", "text3"]},
    timeout=30
)

result = response.json()
embeddings = result["embeddings"]  # List of embedding vectors
dimension = result["dimension"]    # 768
count = result["count"]            # Number of embeddings generated
model = result["model"]            # "Alibaba-NLP/gte-multilingual-base"
```

**Single text request:**

```python
response = httpx.post(
    "https://[your-username]--gte-multilingual-embeddings-embed.modal.run",
    json={"text": "single text here"},
    timeout=30
)
```

### Response Format

```json
{
    "embeddings": [[0.123, -0.456, ...], [0.789, -0.012, ...]],
    "dimension": 768,
    "count": 2,
    "model": "Alibaba-NLP/gte-multilingual-base"
}
```

## Re-Embedding Your Data

After deploying GTE, you'll need to re-embed all hadith data:

### Option 1: Create New Collections

Create new ChromaDB collections with GTE embeddings:

```python
# Update your embedding script to use GTE endpoint
MODAL_EMBED_URL = "https://[your-username]--gte-multilingual-embeddings-embed.modal.run"

# Create new collection names
COLLECTION_NAME = "hadith_bukhari_gte"  # or hadith_muslim_gte
```

### Option 2: Replace Existing Collections

If you want to replace E5 with GTE completely:

1. Backup existing collections
2. Delete old collections
3. Re-run embedding pipeline with GTE endpoint

## Integration with HybridSearchTool

After re-embedding, update your search configuration:

```python
# In your embeddings.py or wherever you configure embeddings
MODAL_EMBED_URL = "https://[your-username]--gte-multilingual-embeddings-embed.modal.run"
EMBEDDING_DIMENSION = 768  # Updated from 1024
```

## Cost Considerations

| Resource                | E5 Large    | GTE Base    |
| ----------------------- | ----------- | ----------- |
| GPU Memory              | ~2GB        | ~1.2GB      |
| Inference Time          | Baseline    | ~20% faster |
| Storage (per embedding) | 1024 floats | 768 floats  |
| Modal GPU Cost          | Same T4     | Same T4     |

## Troubleshooting

### Model Download Fails

```bash
# Clear cache and redeploy
modal volume delete gte-embedding-model-cache
modal deploy zOther/modal_gte_embedding_model.py
```

### Timeout Errors

Increase timeout in your client:

```python
response = httpx.post(url, json=data, timeout=60.0)
```

### Out of Memory

The model uses ~1.2GB GPU memory. T4 (16GB) should handle it easily. If issues persist:

- Reduce batch size
- Check for other processes using GPU memory

## Files Reference

| File                           | Purpose                      |
| ------------------------------ | ---------------------------- |
| `modal_gte_embedding_model.py` | Main Modal deployment script |
| `test_gte_endpoint.py`         | Endpoint testing script      |
| `GTE_DEPLOYMENT.md`            | This documentation           |

## Related Documentation

- [GTE Model on HuggingFace](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- [Modal Documentation](https://modal.com/docs)
- [Sentence Transformers](https://www.sbert.net/)
