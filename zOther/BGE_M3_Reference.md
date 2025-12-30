# BGE-M3 Model Reference

## Overview

BGE-M3 is distinguished for its versatility in **Multi-Functionality**, **Multi-Linguality**, and **Multi-Granularity**.

| Feature                 | Description                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------- |
| **Multi-Functionality** | Simultaneously performs dense retrieval, multi-vector retrieval, and sparse retrieval |
| **Multi-Linguality**    | Supports 100+ languages                                                               |
| **Multi-Granularity**   | Processes inputs from short sentences to long documents (up to 8192 tokens)           |

---

## Retrieval Methods Comparison

| Method               | Description                                                | Example               |
| -------------------- | ---------------------------------------------------------- | --------------------- |
| **Dense Retrieval**  | Maps text into a single embedding                          | DPR, BGE-v1.5         |
| **Sparse Retrieval** | Vector of vocabulary size, weights only for tokens present | BM25, UniCOIL, SPLADE |
| **Multi-Vector**     | Uses multiple vectors to represent text                    | ColBERT               |

---

## Recommended RAG Pipeline

**Hybrid Retrieval + Re-ranking**

1. **Hybrid Retrieval**: Combines embedding retrieval + sparse retrieval (like BM25)

   - BGE-M3 supports both in one model
   - Get token weights (BM25-like) at no extra cost when generating dense embeddings
   - Tools: Vespa, Milvus

2. **Re-ranking**: Use cross-encoder models (e.g., bge-reranker, bge-reranker-v2) after retrieval for higher accuracy

---

## Code Examples

### Dense Embedding

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences_1 = ["What is BGE M3?", "Definition of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval...",
               "BM25 is a bag-of-words retrieval function..."]

embeddings_1 = model.encode(sentences_1,
                            batch_size=12,
                            max_length=8192)['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']

similarity = embeddings_1 @ embeddings_2.T
```

### Sparse Embedding (Lexical Weight)

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
output_2 = model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=False)

# View token weights
print(model.convert_id_to_token(output_1['lexical_weights']))
# [{'What': 0.08356, 'is': 0.0814, 'B': 0.1296, 'GE': 0.252, 'M': 0.1702, '3': 0.2695, '?': 0.04092}, ...]

# Compute lexical matching score
lexical_scores = model.compute_lexical_matching_score(
    output_1['lexical_weights'][0],
    output_2['lexical_weights'][0]
)
```

### Multi-Vector (ColBERT)

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=True)
output_2 = model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=True)

# ColBERT score
print(model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][0]))
# 0.7797
```

### Compute Combined Scores (All Methods)

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentence_pairs = [[i, j] for i in sentences_1 for j in sentences_2]

scores = model.compute_score(
    sentence_pairs,
    max_passage_length=128,  # smaller = lower latency
    weights_for_different_modes=[0.4, 0.2, 0.4]  # dense, sparse, colbert
)

# Returns dict with:
# - 'colbert': [...]
# - 'sparse': [...]
# - 'dense': [...]
# - 'sparse+dense': [...]
# - 'colbert+sparse+dense': [...]
```

---

## Key Parameters

| Parameter                     | Description                                                      | Default            |
| ----------------------------- | ---------------------------------------------------------------- | ------------------ |
| `use_fp16`                    | Use FP16 for faster computation (slight performance degradation) | `True` recommended |
| `batch_size`                  | Batch size for encoding                                          | 12                 |
| `max_length`                  | Maximum token length                                             | 8192               |
| `return_dense`                | Return dense embeddings                                          | `True`             |
| `return_sparse`               | Return sparse/lexical weights                                    | `False`            |
| `return_colbert_vecs`         | Return ColBERT vectors                                           | `False`            |
| `max_passage_length`          | For compute_score - smaller = lower latency                      | 128                |
| `weights_for_different_modes` | Weights for [dense, sparse, colbert] in combined score           | `[0.4, 0.2, 0.4]`  |

---

## Important Notes

1. **No instruction prefix needed**: Unlike BGE, BGE-M3 does NOT require adding instructions to queries
2. **Hybrid retrieval tools**: Vespa, Milvus (pymilvus has example: `hello_hybrid_sparse_dense.py`)
3. **Fine-tuning**: Can fine-tune all embedding functions (dense, sparse, colbert) - see unified_fine-tuning example
4. **Performance**: Achieves top performance in both English and other languages, surpassing OpenAI models

---

## Output Format Reference

### Dense Vectors

- Single vector per text
- Dimension: 1024

### Sparse/Lexical Weights

- Dictionary: `{token: weight, ...}`
- Only non-zero weights for tokens in text
- Use `model.convert_id_to_token()` to see readable tokens

### ColBERT Vectors

- Multiple vectors per text (one per token)
- Used with `model.colbert_score()` for comparison
