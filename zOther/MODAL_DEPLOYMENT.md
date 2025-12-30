# Modal Embedding Model Deployment Guide

This guide explains how to deploy the multilingual-e5-large embedding model to Modal.com.

## Model Information

- **Model**: intfloat/multilingual-e5-large
- **Purpose**: Generate embeddings for Arabic and English hadith texts
- **Embedding Dimension**: 1024
- **Max Sequence Length**: 512 tokens
- **Supports**: 100+ languages including Arabic and English

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

### 2. Deploy the Embedding Model

Deploy the model to Modal:

```bash
modal deploy zOther/modal_embedding_model.py
```

This will:

- Create a Modal app called "multilingual-e5-embeddings"
- Download the model from Hugging Face (~2GB)
- Cache it in a Modal Volume for reuse
- Deploy a web endpoint for API access

**Expected output:**

```
✓ Created objects.
├── 🔨 Created mount /path/to/project
├── 🔨 Created download_model.
├── 🔨 Created EmbeddingModel.
├── 🔨 Created test_embeddings.
└── 🔨 Created web function embed => https://[your-username]--multilingual-e5-embeddings-embed.modal.run
✓ App deployed! 🎉
```

### 3. Copy Your Endpoint URL

After successful deployment, Modal will provide a URL like:

```
https://[your-username]--multilingual-e5-embeddings-embed.modal.run
```

**Copy this URL** - you'll need it in the next step.

### 4. Update the Embedding Script

Open `src/data/embed_chunks.py` and update line 36 with your URL:

```python
MODAL_EMBED_URL = "https://[your-username]--multilingual-e5-embeddings-embed.modal.run"
```

Replace `[your-username]` with your actual Modal username.

### 5. Test the Deployment

Test the model with sample texts:

```bash
modal run zOther/modal_embedding_model.py
```

This will:

- Generate embeddings for Arabic and English test texts
- Display embedding dimensions
- Calculate similarity between translations

**Expected output:**

```
Testing embedding generation...

Generated 3 embeddings
Embedding dimension: 1024
First embedding (first 10 values): [0.123, -0.456, ...]

Cosine similarity between Arabic and English: 0.8543

Test completed successfully!
```

### 6. Run the Embedding Pipeline

Now you can generate embeddings for all hadiths:

```bash
python src/data/embed_chunks.py
```

This will:

- Process ~33,000 chunks (Bukhari + Muslim)
- Generate embeddings in batches of 50
- Store embeddings in ChromaDB

## API Usage

### Request Format

**Batch request:**

```python
import httpx

response = httpx.post(
    "https://[your-username]--multilingual-e5-embeddings-embed.modal.run",
    json={"texts": ["text1", "text2", "text3"]},
    timeout=30
)

result = response.json()
embeddings = result["embeddings"]  # List of embedding vectors
dimension = result["dimension"]    # 1024
count = result["count"]           # Number of embeddings generated
```

**Single text request:**

```python
response = httpx.post(
    "https://[your-username]--multilingual-e5-embeddings-embed.modal.run",
    json={"text": "single text here"},
    timeout=30
)

result = response.json()
embedding = result["embeddings"][0]  # Single embedding vector
```

## Cost Considerations

Modal pricing is based on GPU time used. The script is configured to:

- **GPU**: T4 (cost-effective for embeddings)
- **Keep Warm**: 1 container (reduces cold starts)
- **Idle Timeout**: 5 minutes
- **Batch Size**: 50 texts per request (optimal for throughput)

For processing ~33,000 chunks:

- Estimated time: 15-30 minutes
- Estimated cost: $1-2 (varies by region)

## Monitoring

View your deployment status and logs:

```bash
modal app list
modal app logs multilingual-e5-embeddings
```

## Troubleshooting

### Error: "No text provided"

**Cause**: Request body is missing `text` or `texts` key

**Solution**: Use the correct request format:

```python
{"texts": ["text1", "text2"]}  # Batch
# or
{"text": "single text"}        # Single
```

### Error: "Model not found"

**Cause**: Model download failed or volume not mounted

**Solution**: Re-run the download:

```bash
modal run zOther/modal_embedding_model.py --action download
```

### Slow Initial Response

**Cause**: Cold start - container is initializing

**Solution**: The `keep_warm=1` setting keeps one container warm. First request after idle timeout will be slower.

### Out of Memory

**Cause**: Batch size too large

**Solution**: Reduce `BATCH_SIZE` in `src/data/embed_chunks.py` from 50 to 25 or 10.

## Updating the Model

To update or redeploy:

```bash
# Make changes to modal_embedding_model.py
modal deploy zOther/modal_embedding_model.py --force
```

## Cleaning Up

To remove the deployment:

```bash
modal app stop multilingual-e5-embeddings
```

To remove cached model volumes:

```bash
modal volume delete embedding-model-cache
modal volume delete huggingface-cache
```

## Next Steps

After successful embedding generation:

1. **Verify ChromaDB**: Check that embeddings are stored
2. **Test Retrieval**: Query the vector database
3. **Build Agents**: Implement the LangGraph workflow
4. **Deploy API**: Create FastAPI endpoints for queries

## Support

- Modal Docs: https://modal.com/docs
- Model Card: https://huggingface.co/intfloat/multilingual-e5-large
- Project Issues: [Your GitHub repo]
