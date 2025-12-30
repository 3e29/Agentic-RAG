"""
Quick test script to verify the Modal embedding endpoint works correctly.
"""
import httpx
import json

MODAL_EMBED_URL = "https://sazaitet110--multilingual-e5-embeddings-embed.modal.run"

def test_single_text():
    """Test single text embedding."""
    print("\n=== Testing Single Text ===")
    
    response = httpx.post(
        MODAL_EMBED_URL,
        json={"text": "بسم الله الرحمن الرحيم"},
        timeout=30.0
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response keys: {data.keys()}")
    print(f"Embedding dimension: {data.get('dimension')}")
    print(f"Number of embeddings: {data.get('count')}")
    print(f"First 5 values: {data['embeddings'][0][:5]}")
    
    return response.status_code == 200


def test_batch_texts():
    """Test batch text embedding."""
    print("\n=== Testing Batch Texts ===")
    
    test_texts = [
        "بسم الله الرحمن الرحيم",
        "In the name of Allah, the Most Gracious, the Most Merciful",
        "حدثنا محمد بن إسماعيل البخاري",
        "The Prophet Muhammad (peace be upon him) said",
        "الحمد لله رب العالمين"
    ]
    
    response = httpx.post(
        MODAL_EMBED_URL,
        json={"texts": test_texts},
        timeout=30.0
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response keys: {data.keys()}")
    print(f"Embedding dimension: {data.get('dimension')}")
    print(f"Number of embeddings: {data.get('count')}")
    print(f"Expected embeddings: {len(test_texts)}")
    
    # Verify we got the right number of embeddings
    assert data['count'] == len(test_texts), f"Expected {len(test_texts)} embeddings, got {data['count']}"
    print(f"✓ Got correct number of embeddings!")
    
    return response.status_code == 200


def test_large_batch():
    """Test with 50 texts (typical batch size)."""
    print("\n=== Testing Large Batch (50 texts) ===")
    
    test_texts = [f"Test hadith text number {i} في الحديث" for i in range(50)]
    
    response = httpx.post(
        MODAL_EMBED_URL,
        json={"texts": test_texts},
        timeout=60.0
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Embedding dimension: {data.get('dimension')}")
    print(f"Number of embeddings: {data.get('count')}")
    print(f"Expected embeddings: {len(test_texts)}")
    
    # Verify we got the right number of embeddings
    assert data['count'] == len(test_texts), f"Expected {len(test_texts)} embeddings, got {data['count']}"
    print(f"✓ Got correct number of embeddings!")
    
    return response.status_code == 200


if __name__ == "__main__":
    print(f"Testing Modal Embedding Endpoint: {MODAL_EMBED_URL}")
    
    try:
        # Run all tests
        result1 = test_single_text()
        result2 = test_batch_texts()
        result3 = test_large_batch()
        
        if result1 and result2 and result3:
            print("\n" + "="*60)
            print("✓ All tests passed! Endpoint is working correctly.")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("✗ Some tests failed. Check the output above.")
            print("="*60)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
