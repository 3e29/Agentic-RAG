"""
Modal Deployment Script for Multilingual E5 Embedding Model

This script deploys the intfloat/multilingual-e5-large model to Modal
for generating embeddings for Arabic and English hadith texts.

Model: intfloat/multilingual-e5-large
- Supports 100+ languages including Arabic
- 512 token context length
- Optimized for semantic similarity tasks

Usage:
    modal deploy zOther/modal_embedding_model.py
"""

from pathlib import Path
from typing import List

import modal

# Configuration
MODEL_NAME = "intfloat/multilingual-e5-large"
MODEL_DIR = "/models"
CACHE_DIR = "/cache"

# Create Modal volumes for model caching
model_volume = modal.Volume.from_name("embedding-model-cache", create_if_missing=True)
cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Create Modal app
app = modal.App("multilingual-e5-embeddings")

# Define the image with all dependencies
embedding_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers==3.3.1",
        "transformers==4.48.1",
        "torch==2.5.1",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": CACHE_DIR,
        "TRANSFORMERS_CACHE": CACHE_DIR,
    })
)


def download_model_func():
    """
    Download the multilingual-e5-large model from Hugging Face.
    This runs once during image build to cache the model.
    """
    from sentence_transformers import SentenceTransformer
    
    print(f"Downloading model: {MODEL_NAME}")
    
    # Download and cache the model
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_DIR)
    
    print(f"Model downloaded successfully to {MODEL_DIR}")
    print(f"Model max sequence length: {model.max_seq_length}")


# Build the inference image with the downloaded model
inference_image = (
    embedding_image
    .run_function(
        download_model_func,
        volumes={
            MODEL_DIR: model_volume,
            CACHE_DIR: cache_volume,
        }
    )
)


@app.cls(
    image=inference_image,
    gpu="T4",  # T4 GPU for fast inference
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
    min_containers=1,  # Keep 1 instance warm for low latency
    scaledown_window=300,  # 5 minutes idle timeout
)
class EmbeddingModel:
    """
    Embedding model class for generating embeddings via API.
    """
    
    @modal.enter()
    def load_model(self):
        """Load the model when the container starts."""
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading model from {MODEL_DIR}")
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_DIR)
        print(f"Model loaded successfully")
        print(f"Max sequence length: {self.model.max_seq_length}")
    
    @modal.method()
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        import numpy as np
        
        if not texts:
            return []
        
        # Add instruction prefix for better performance (recommended by model authors)
        # For retrieval tasks, prefix queries with "query: " and passages with "passage: "
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        
        # Convert numpy arrays to lists for JSON serialization
        return embeddings.tolist()
    
    @modal.method()
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        result = self.embed_texts([text])
        return result[0] if result else []


@app.function(
    image=inference_image,
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
)
def test_embeddings():
    """
    Test the embedding model with sample Arabic and English texts.
    """
    model_instance = EmbeddingModel()
    
    # Test texts
    test_texts = [
        "بسم الله الرحمن الرحيم",  # Arabic
        "In the name of Allah, the Most Gracious, the Most Merciful",  # English
        "حدثنا محمد بن إسماعيل البخاري",  # Arabic hadith style
    ]
    
    print("\nTesting embedding generation...")
    embeddings = model_instance.embed_texts.remote(test_texts)
    
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"First embedding (first 10 values): {embeddings[0][:10]}")
    
    # Test cosine similarity between Arabic and English versions
    import numpy as np
    from numpy.linalg import norm
    
    emb1 = np.array(embeddings[0])
    emb2 = np.array(embeddings[1])
    
    # Cosine similarity (already normalized, so just dot product)
    similarity = np.dot(emb1, emb2)
    print(f"\nCosine similarity between Arabic and English: {similarity:.4f}")
    
    print("\nTest completed successfully!")


# Web endpoint for external API calls
@app.function(
    image=inference_image,
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
    min_containers=1,
)
@modal.fastapi_endpoint(method="POST")
def embed(request: dict) -> dict:
    """
    Web endpoint for generating embeddings.
    
    Request format:
        {"texts": ["text1", "text2", ...]}
        or
        {"text": "single text"}
    
    Response format:
        {"embeddings": [[...], [...], ...], "dimension": 1024}
    """
    model_instance = EmbeddingModel()
    
    # Handle both single text and batch requests
    if "texts" in request:
        texts = request["texts"]
    elif "text" in request:
        texts = [request["text"]]
    else:
        return {"error": "No text provided. Use 'text' or 'texts' key."}, 400
    
    if not texts:
        return {"error": "Empty text list"}, 400
    
    # Generate embeddings
    try:
        embeddings = model_instance.embed_texts.remote(texts)
        return {
            "embeddings": embeddings,
            "dimension": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings),
        }
    except Exception as e:
        return {"error": str(e)}, 500


# CLI commands
@app.local_entrypoint()
def main(action: str = "test"):
    """
    Local entrypoint for testing and management.
    
    Args:
        action: Action to perform (test)
    """
    if action == "test":
        print("Running embedding tests...")
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

3. Deploy the embedding model:
   modal deploy zOther/modal_embedding_model.py

4. Test the deployment:
   modal run zOther/modal_embedding_model.py

5. Get the web endpoint URL:
   After deployment, Modal will provide a URL like:
   https://[your-username]--multilingual-e5-embeddings-embed.modal.run

6. Use the endpoint in your embedding script:
   Update MODAL_EMBED_URL in src/data/embed_chunks.py

Example API usage:
==================

import httpx

response = httpx.post(
    "https://[your-username]--multilingual-e5-embeddings-embed.modal.run",
    json={"texts": ["text1", "text2"]},
    timeout=30
)
embeddings = response.json()["embeddings"]
"""
