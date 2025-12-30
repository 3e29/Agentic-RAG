"""
Quick test script to verify the Modal GTE embedding endpoint works correctly.

Model: Alibaba-NLP/gte-multilingual-base
- 768 embedding dimensions
- 8192 max tokens
- 70+ languages including Arabic
"""
import httpx
import json
import numpy as np

# UPDATE THIS URL after deploying to Modal
MODAL_GTE_URL = "https://sazaitet110--gte-multilingual-embeddings-embed.modal.run"
 


def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(emb1, emb2)  # Already normalized


def test_single_text():
    """Test single text embedding."""
    print("\n=== Testing Single Text ===")
    
    response = httpx.post(
        MODAL_GTE_URL,
        json={"text": "بسم الله الرحمن الرحيم"},
        timeout=30.0
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response keys: {data.keys()}")
    print(f"Embedding dimension: {data.get('dimension')}")
    print(f"Number of embeddings: {data.get('count')}")
    print(f"Model: {data.get('model')}")
    print(f"First 5 values: {data['embeddings'][0][:5]}")
    
    # Verify dimension is 768 (GTE model)
    assert data['dimension'] == 768, f"Expected 768 dimensions (GTE), got {data['dimension']}"
    print(f"✓ Correct embedding dimension (768)!")
    
    return response.status_code == 200


def test_batch_texts():
    """Test batch text embedding with multilingual content."""
    print("\n=== Testing Batch Texts (Multilingual) ===")
    
    test_texts = [
        "بسم الله الرحمن الرحيم",
        "In the name of Allah, the Most Gracious, the Most Merciful",
        "حدثنا محمد بن إسماعيل البخاري",
        "The Prophet Muhammad (peace be upon him) said",
        "الحمد لله رب العالمين"
    ]
    
    response = httpx.post(
        MODAL_GTE_URL,
        json={"texts": test_texts},
        timeout=30.0
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Embedding dimension: {data.get('dimension')}")
    print(f"Number of embeddings: {data.get('count')}")
    print(f"Expected embeddings: {len(test_texts)}")
    
    # Verify we got the right number of embeddings
    assert data['count'] == len(test_texts), f"Expected {len(test_texts)} embeddings, got {data['count']}"
    print(f"✓ Got correct number of embeddings!")
    
    # Verify dimension
    assert data['dimension'] == 768, f"Expected 768 dimensions, got {data['dimension']}"
    print(f"✓ Correct embedding dimension!")
    
    return response.status_code == 200


def test_crosslingual_similarity():
    """Test cross-lingual similarity between Arabic and English."""
    print("\n=== Testing Cross-Lingual Similarity ===")
    
    test_pairs = [
        {
            "arabic": "بسم الله الرحمن الرحيم",
            "english": "In the name of Allah, the Most Gracious, the Most Merciful",
            "description": "Bismillah"
        },
        {
            "arabic": "الصبر نصف الإيمان",
            "english": "Patience is half of faith",
            "description": "Patience/Faith"
        },
        {
            "arabic": "الحمد لله رب العالمين",
            "english": "Praise be to Allah, Lord of all the worlds",
            "description": "Alhamdulillah"
        }
    ]
    
    for pair in test_pairs:
        response = httpx.post(
            MODAL_GTE_URL,
            json={"texts": [pair["arabic"], pair["english"]]},
            timeout=30.0
        )
        
        data = response.json()
        emb_arabic = np.array(data['embeddings'][0])
        emb_english = np.array(data['embeddings'][1])
        
        similarity = cosine_similarity(emb_arabic, emb_english)
        print(f"  {pair['description']}: Arabic ↔ English similarity = {similarity:.4f}")
    
    print(f"✓ Cross-lingual similarity test completed!")
    return True


def test_large_batch():
    """Test with 50 texts (typical batch size)."""
    print("\n=== Testing Large Batch (50 texts) ===")
    
    test_texts = [f"Test hadith text number {i} في الحديث رقم {i}" for i in range(50)]
    
    response = httpx.post(
        MODAL_GTE_URL,
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


def test_long_text():
    """Test with longer text (GTE supports 8192 tokens)."""
    print("\n=== Testing Long Text (GTE 8192 token advantage) ===")
    
    # Create a longer hadith-style text
    long_text = """
    حدثنا محمد بن إسماعيل البخاري قال حدثنا عبد الله بن يوسف قال أخبرنا مالك عن نافع عن 
    عبد الله بن عمر رضي الله عنهما أن رسول الله صلى الله عليه وسلم قال بني الإسلام على خمس 
    شهادة أن لا إله إلا الله وأن محمدا رسول الله وإقام الصلاة وإيتاء الزكاة والحج وصوم رمضان.
    
    وفي رواية أخرى عن ابن عمر قال قال رسول الله صلى الله عليه وسلم الإسلام أن تشهد أن لا إله 
    إلا الله وأن محمدا رسول الله وتقيم الصلاة وتؤتي الزكاة وتصوم رمضان وتحج البيت إن استطعت 
    إليه سبيلا والإيمان أن تؤمن بالله وملائكته وكتبه ورسله واليوم الآخر وتؤمن بالقدر خيره وشره.
    """ * 5  # Repeat to make it longer
    
    response = httpx.post(
        MODAL_GTE_URL,
        json={"text": long_text},
        timeout=30.0
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Text length: {len(long_text)} characters")
    print(f"Embedding dimension: {data.get('dimension')}")
    print(f"✓ Successfully embedded long text!")
    
    return response.status_code == 200


def compare_with_e5():
    """Compare GTE with E5 (if E5 endpoint is available)."""
    print("\n=== Comparing GTE vs E5 (Optional) ===")
    
    E5_URL = "https://sazaitet110--multilingual-e5-embeddings-embed.modal.run"
    
    test_text = "الصبر والشكر من أعظم أعمال القلوب"
    
    try:
        # Get GTE embedding
        gte_response = httpx.post(
            MODAL_GTE_URL,
            json={"text": test_text},
            timeout=30.0
        )
        gte_dim = gte_response.json()['dimension']
        
        # Get E5 embedding (if available)
        e5_response = httpx.post(
            E5_URL,
            json={"text": test_text},
            timeout=30.0
        )
        e5_dim = e5_response.json()['dimension']
        
        print(f"  GTE dimension: {gte_dim}")
        print(f"  E5 dimension: {e5_dim}")
        print(f"  GTE max tokens: 8192")
        print(f"  E5 max tokens: 512")
        print(f"✓ Both models working!")
        
    except Exception as e:
        print(f"  E5 comparison skipped (endpoint not available)")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("Testing Modal GTE Embedding Endpoint")
    print(f"URL: {MODAL_GTE_URL}")
    print("="*60)
    
    if "YOUR-USERNAME" in MODAL_GTE_URL:
        print("\n⚠️  WARNING: Update MODAL_GTE_URL with your actual Modal endpoint URL")
        print("   After deploying with: modal deploy zOther/modal_gte_embedding_model.py")
        print("   Modal will provide your endpoint URL")
        exit(1)
    
    try:
        # Run all tests
        result1 = test_single_text()
        result2 = test_batch_texts()
        result3 = test_crosslingual_similarity()
        result4 = test_large_batch()
        result5 = test_long_text()
        compare_with_e5()
        
        if result1 and result2 and result3 and result4 and result5:
            print("\n" + "="*60)
            print("✓ All tests passed! GTE endpoint is working correctly.")
            print("="*60)
            print("\nGTE Model Advantages:")
            print("  • 8192 max tokens (vs E5's 512)")
            print("  • No instruction prefix required")
            print("  • 768 dimensions (more efficient than E5's 1024)")
            print("  • Strong multilingual + Arabic support")
        else:
            print("\n" + "="*60)
            print("✗ Some tests failed. Check the output above.")
            print("="*60)
            
    except httpx.ConnectError:
        print(f"\n✗ Could not connect to {MODAL_GTE_URL}")
        print("  Make sure the endpoint is deployed and the URL is correct.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
