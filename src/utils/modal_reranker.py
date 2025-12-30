"""
Modal Deployment for BGE Reranker v2 M3

Deploys BAAI/bge-reranker-v2-m3 on Modal for:
- GPU-accelerated inference
- No local model download (2.27GB)
- Scalable serverless deployment

Usage:
    # Deploy to Modal
    modal deploy src/utils/modal_reranker.py
    
    # Test locally
    modal run src/utils/modal_reranker.py::test_reranker
"""

import modal

# ============================================================================
# Modal Configuration
# ============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
    )
)

app = modal.App("hadith-reranker")

# Model volume for caching
model_volume = modal.Volume.from_name("reranker-model-cache", create_if_missing=True)
MODEL_CACHE_PATH = "/root/model-cache"


# ============================================================================
# Reranker Service
# ============================================================================

@app.cls(
    gpu="T4",  # T4 is cost-effective for inference
    image=image,
    volumes={MODEL_CACHE_PATH: model_volume},
    scaledown_window=300,  # Keep warm for 5 minutes
    timeout=600,  # 10 minute timeout for long batches
)
class Reranker:
    """
    BGE Reranker v2 M3 service on Modal.
    
    Provides multilingual cross-encoder reranking with Arabic support.
    """
    
    @modal.enter()
    def load_model(self):
        """Load the reranker model on container startup."""
        import os
        from sentence_transformers import CrossEncoder
        
        # Set cache directory
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_PATH
        os.environ["HF_HOME"] = MODEL_CACHE_PATH
        
        print("Loading BGE Reranker v2 M3...")
        self.model = CrossEncoder(
            'BAAI/bge-reranker-v2-m3',
            max_length=512,
            device='cuda'
        )
        print("Reranker model loaded successfully!")
        
        # Commit the cached model to volume
        model_volume.commit()
    
    @modal.method()
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """
        Rerank passages by relevance to query.
        
        Args:
            query: The search query
            passages: List of passage texts to rerank
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not passages:
            return []
        
        # Create query-passage pairs
        pairs = [[query, p] for p in passages]
        
        # Get relevance scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        return scores.tolist()
    
    @modal.method()
    def rerank_batch(
        self, 
        queries: list[str], 
        passages_list: list[list[str]]
    ) -> list[list[float]]:
        """
        Batch reranking for multiple queries.
        
        Args:
            queries: List of search queries
            passages_list: List of passage lists (one per query)
            
        Returns:
            List of score lists (one per query)
        """
        results = []
        for query, passages in zip(queries, passages_list):
            if passages:
                pairs = [[query, p] for p in passages]
                scores = self.model.predict(pairs, show_progress_bar=False)
                results.append(scores.tolist())
            else:
                results.append([])
        return results
    
    @modal.method()
    def health_check(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": "BAAI/bge-reranker-v2-m3",
            "device": "cuda"
        }


# ============================================================================
# Test Function
# ============================================================================

@app.function(image=image)
def test_reranker():
    """Test the reranker service."""
    reranker = Reranker()
    
    # Test Arabic query
    query = "بدء الوحي نزول جبريل غار حراء"
    passages = [
        "حدّثنا يحيى بن بكير قال حدّثنا الليث عن عقيل",  # Just isnad
        "أول ما بدئ به رسول الله من الوحي الرؤيا الصالحة في النوم فكان لا يرى رؤيا إلا جاءت مثل فلق الصبح",  # Revelation hadith
        "كتاب النكاح باب الزواج",  # Unrelated
    ]
    
    print(f"Query: {query}")
    print(f"\nPassages:")
    for i, p in enumerate(passages):
        print(f"  {i+1}. {p[:50]}...")
    
    scores = reranker.rerank.remote(query, passages)
    
    print(f"\nScores:")
    for i, (p, s) in enumerate(zip(passages, scores)):
        print(f"  {i+1}. Score: {s:.4f} - {p[:40]}...")
    
    # Verify the revelation hadith (index 1) has highest score
    assert scores[1] > scores[0], "Revelation hadith should rank higher than isnad"
    assert scores[1] > scores[2], "Revelation hadith should rank higher than unrelated"
    
    print("\n✓ Test passed! Reranker correctly ranks revelation hadith highest.")
    return scores


# ============================================================================
# Local Entry Point
# ============================================================================

if __name__ == "__main__":
    # For local testing with modal run
    with app.run():
        test_reranker.remote()
