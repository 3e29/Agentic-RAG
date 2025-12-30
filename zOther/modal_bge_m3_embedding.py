"""
Modal Deployment Script for BGE-M3 Embedding Model

This script deploys the BAAI/bge-m3 model to Modal for generating 
multi-vector embeddings (Dense + Sparse) for Arabic and English hadith texts.

Model: BAAI/bge-m3
- Supports 100+ languages with excellent Arabic performance
- 8192 token context length
- 1024 embedding dimensions
- Multi-Vector: Dense + Sparse + ColBERT representations
- Best-in-class for Arabic retrieval (MIRACL benchmark)

Key Features:
- Dense vectors for semantic search
- Sparse vectors for keyword matching (like BM25 but learned)
- Can replace BM25 + Vector search with a single model!

Usage:
    modal deploy zOther/modal_bge_m3_embedding.py
"""

from pathlib import Path
from typing import List, Dict, Any

import modal
#569M parameters
#BAAI/bge-reranker-v2-m3 0.6B parameters
# Configuration
MODEL_NAME = "BAAI/bge-m3"
MODEL_DIR = "/models"
CACHE_DIR = "/cache"


# Create Modal volumes for model caching
model_volume = modal.Volume.from_name("bge-m3-model-cache", create_if_missing=True)
cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Create Modal app
app = modal.App("bge-m3-embeddings")

# Define the image with all dependencies
embedding_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "FlagEmbedding>=1.2.10",
        "transformers>=4.45.0,<4.49.0",
        "torch==2.5.1",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
        "peft>=0.13.0,<0.15.0",  # Pin compatible peft version
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": CACHE_DIR,
        "TRANSFORMERS_CACHE": CACHE_DIR,
    })
)


def download_model_func():
    """
    Download the BGE-M3 model from Hugging Face.
    This runs once during image build to cache the model.
    """
    from FlagEmbedding import BGEM3FlagModel
    
    print(f"Downloading model: {MODEL_NAME}")
    
    # Download and cache the model
    model = BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=True,
    )
    
    print(f"Model downloaded successfully")
    print(f"Model supports: Dense, Sparse, and ColBERT representations")


# Build the inference image with the downloaded model
inference_image = (
    embedding_image
    .run_function(
        download_model_func,
        volumes={
            MODEL_DIR: model_volume,
            CACHE_DIR: cache_volume,
        },
        timeout=1800,  # 30 min timeout for model download
    )
)


@app.cls(
    image=inference_image,
    gpu="T4",  # T4 GPU for fast inference
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
    min_containers=0,  # Scale to zero when not in use
    scaledown_window=300,  # 5 minutes idle timeout
)
class BGEM3Model:
    """
    BGE-M3 Embedding model class for generating multi-vector embeddings.
    
    Supports three types of representations:
    1. Dense (1024-dim): For semantic similarity
    2. Sparse: For keyword matching (learned BM25-like)
    3. ColBERT: For fine-grained token-level matching
    """
    
    @modal.enter()
    def load_model(self):
        """Load the model when the container starts."""
        from FlagEmbedding import BGEM3FlagModel
        
        print(f"Loading BGE-M3 model...")
        self.model = BGEM3FlagModel(
            MODEL_NAME,
            use_fp16=True,
        )
        print(f"Model loaded successfully")
    
    @modal.method()
    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings (1024-dim) for semantic search.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of dense embedding vectors (each is 1024 floats)
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        
        # Extract dense vectors and convert to list
        dense_vecs = embeddings['dense_vecs']
        return dense_vecs.tolist()
    
    @modal.method()
    def embed_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Generate sparse embeddings for keyword matching.
        
        This is like a learned BM25 - captures term importance.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of sparse vectors (dict of token_id -> weight)
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            max_length=8192,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        
        # Convert sparse format - lexical_weights is already a list of dicts
        sparse_vecs = embeddings['lexical_weights']
        
        # Convert to serializable format
        result = []
        for vec in sparse_vecs:
            # vec is a dict with token_id (int) -> weight (float)
            result.append({str(k): float(v) for k, v in vec.items()})
        
        return result
    
    @modal.method()
    def embed_multi(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate both dense and sparse embeddings in one pass.
        
        This is the recommended method for hybrid search.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Dict with 'dense' and 'sparse' embeddings
        """
        if not texts:
            return {"dense": [], "sparse": []}
        
        # Generate both types of embeddings in one pass
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        
        # Extract dense vectors
        dense_vecs = embeddings['dense_vecs'].tolist()
        
        # Extract sparse vectors
        sparse_vecs = []
        for vec in embeddings['lexical_weights']:
            sparse_vecs.append({str(k): float(v) for k, v in vec.items()})
        
        return {
            "dense": dense_vecs,
            "sparse": sparse_vecs,
        }


@app.function(
    image=inference_image,
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
)
def test_embeddings():
    """
    Test the BGE-M3 model with sample Arabic and English texts.
    """
    model_instance = BGEM3Model()
    
    # Test texts - mix of Arabic and English
    test_texts = [
        "بسم الله الرحمن الرحيم",  # Arabic: Bismillah
        "In the name of Allah, the Most Gracious, the Most Merciful",  # English
        "الصبر والشكر من أعظم أعمال القلوب",  # Arabic: Patience and gratitude
        "Patience and gratitude are among the greatest deeds of the heart",  # English
        "حدثنا محمد بن إسماعيل البخاري",  # Arabic: Hadith chain
    ]
    
    print("\nTesting BGE-M3 embedding generation...")
    
    # Test dense embeddings
    print("\n1. Testing Dense Embeddings:")
    dense = model_instance.embed_dense.remote(test_texts)
    print(f"   Generated {len(dense)} dense embeddings")
    print(f"   Dimension: {len(dense[0])}")
    print(f"   First 5 values of first embedding: {dense[0][:5]}")
    
    # Test sparse embeddings
    print("\n2. Testing Sparse Embeddings:")
    sparse = model_instance.embed_sparse.remote(test_texts)
    print(f"   Generated {len(sparse)} sparse embeddings")
    print(f"   Non-zero terms in first embedding: {len(sparse[0])}")
    
    # Test multi-vector embeddings
    print("\n3. Testing Multi-Vector Embeddings:")
    multi = model_instance.embed_multi.remote(test_texts)
    print(f"   Dense vectors: {len(multi['dense'])}")
    print(f"   Sparse vectors: {len(multi['sparse'])}")
    
    # Test cross-lingual similarity
    import numpy as np
    print("\n4. Cross-lingual Similarity (Dense):")
    
    # Bismillah Arabic vs English
    emb1 = np.array(dense[0])
    emb2 = np.array(dense[1])
    sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"   Bismillah Arabic ↔ English: {sim1:.4f}")
    
    # Patience/Gratitude Arabic vs English
    emb3 = np.array(dense[2])
    emb4 = np.array(dense[3])
    sim2 = np.dot(emb3, emb4) / (np.linalg.norm(emb3) * np.linalg.norm(emb4))
    print(f"   Patience/Gratitude Arabic ↔ English: {sim2:.4f}")
    
    print("\nTest completed successfully!")


# Web endpoint for external API calls - Dense only (most common)
@app.function(
    image=inference_image,
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
    min_containers=0,
)
@modal.fastapi_endpoint(method="POST")
def embed(request: dict) -> dict:
    """
    Web endpoint for generating dense embeddings.
    
    Request format:
        {"texts": ["text1", "text2", ...]}
        or
        {"text": "single text"}
    
    Response format:
        {"embeddings": [[...], [...], ...], "dimension": 1024, "count": N}
    """
    model_instance = BGEM3Model()
    
    # Handle both single text and batch requests
    if "texts" in request:
        texts = request["texts"]
    elif "text" in request:
        texts = [request["text"]]
    else:
        return {"error": "No text provided. Use 'text' or 'texts' key."}
    
    if not texts:
        return {"error": "Empty text list"}
    
    # Generate dense embeddings
    try:
        embeddings = model_instance.embed_dense.remote(texts)
        return {
            "embeddings": embeddings,
            "dimension": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings),
            "model": MODEL_NAME,
            "type": "dense",
        }
    except Exception as e:
        return {"error": str(e)}


# Web endpoint for multi-vector embeddings (dense + sparse)
@app.function(
    image=inference_image,
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
    min_containers=0,
)
@modal.fastapi_endpoint(method="POST")
def embed_multi_endpoint(request: dict) -> dict:
    """
    Web endpoint for generating both dense and sparse embeddings.
    
    Request format:
        {"texts": ["text1", "text2", ...]}
    
    Response format:
        {
            "dense": [[...], [...], ...],
            "sparse": [{token_id: weight, ...}, ...],
            "dimension": 1024,
            "count": N
        }
    """
    model_instance = BGEM3Model()
    
    # Handle both single text and batch requests
    if "texts" in request:
        texts = request["texts"]
    elif "text" in request:
        texts = [request["text"]]
    else:
        return {"error": "No text provided. Use 'text' or 'texts' key."}
    
    if not texts:
        return {"error": "Empty text list"}
    
    # Generate multi-vector embeddings
    try:
        result = model_instance.embed_multi.remote(texts)
        return {
            "dense": result["dense"],
            "sparse": result["sparse"],
            "dimension": len(result["dense"][0]) if result["dense"] else 0,
            "count": len(result["dense"]),
            "model": MODEL_NAME,
        }
    except Exception as e:
        return {"error": str(e)}


# CLI commands
@app.local_entrypoint()
def main(action: str = "test"):
    """
    Local entrypoint for testing and management.
    
    Args:
        action: Action to perform (test)
    """
    if action == "test":
        print("Running BGE-M3 embedding tests...")
        test_embeddings.remote()
    else:
        print(f"Unknown action: {action}")
        print("Available actions: test")


"""
Deployment Instructions:
========================

1. Install Modal CLI:
   pip install modal

2. Setup Modal token:
   modal token new

3. Deploy the BGE-M3 embedding model:
   modal deploy zOther/modal_bge_m3_embedding.py

4. Test the deployment:
   modal run zOther/modal_bge_m3_embedding.py

5. Get the web endpoint URLs:
   After deployment, Modal will provide URLs like:
   - Dense: https://[your-username]--bge-m3-embeddings-embed.modal.run
   - Multi: https://[your-username]--bge-m3-embeddings-embed-multi-endpoint.modal.run

BGE-M3 Advantages:
==================
- Multi-Vector: Dense (semantic) + Sparse (keyword) in one model
- Best Arabic performance on MIRACL benchmark
- 8192 token context (vs E5's 512)
- Can replace BM25 + Vector search with single model
- 1024-dim dense vectors

Example API usage:
==================

# Dense only (fast, for semantic search)
import httpx

response = httpx.post(
    "https://[your-username]--bge-m3-embeddings-embed.modal.run",
    json={"texts": ["text1", "text2"]},
    timeout=30
)
embeddings = response.json()["embeddings"]

# Multi-vector (dense + sparse for hybrid search)
response = httpx.post(
    "https://[your-username]--bge-m3-embeddings-embed-multi-endpoint.modal.run",
    json={"texts": ["text1", "text2"]},
    timeout=30
)
result = response.json()
dense = result["dense"]
sparse = result["sparse"]
"""
