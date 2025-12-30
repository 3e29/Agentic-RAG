# Implementation Plan: Hadith RAG System

This document outlines the phased implementation plan for the Hadith RAG system, based on the requirements defined in `initial-requirements.md`.

---

## Phase 1 & 2: Data Foundation (Current Phase)

### Objective

Build a robust data ingestion pipeline that transforms raw Hadith JSON data into a searchable vector database.

### Requirements Covered

- **FR-DIC-31:** Data Ingestion Pipeline
- **FR-DIC-32:** JSON Parsing
- **FR-DIC-33:** Text Preprocessing (Arabic normalization)
- **FR-DIC-34:** Embedding Generation
- **FR-DIC-35:** Vector Storage (ChromaDB)
- **NFR-05:** Data Consistency

### Implementation Steps

#### 1. Consolidate Arabic Text Processing (`src/utils/arabic_processing.py`)

**Status:** Not Started

Consolidate the preprocessing logic from `script.py`, `script2.py`, and `script3.py` into a single, reusable module.

**Functions to Implement:**

- `remove_diacritics_keep_shadda(text: str) -> str`: Remove Arabic diacritics except shadda (ّ)
- `remove_directional_marks(text: str) -> str`: Remove LTR/RTL marks (\u200e, \u200f)
- `normalize_whitespace(text: str) -> str`: Normalize multiple spaces, newlines
- `clean_arabic_text(text: str) -> str`: Main function that applies all preprocessing steps

**Dependencies:** Standard library only (no external packages needed)

#### 2. Build Data Ingestion Pipeline (`src/data/ingestion.py`)

**Status:** Not Started

Create a standalone module for loading, processing, and storing Hadith data.

**Key Components:**

a. **JSON Parser:**

- Load `data/raw/bukhari.json`
- Extract hadith records with metadata (book, chapter, hadith number, narrator)
- Handle both Arabic and English text

b. **Embedding Generator:**

- Call Modal embedding endpoint: `https://sazaitet110--qwen2-5-14b-runner-qwenmodel-embed.modal.run`
- Use `httpx` for async HTTP calls
- Batch processing (recommended: 50-100 hadiths per batch)
- Retry logic for failed requests

c. **ChromaDB Integration:**

- Initialize `PersistentClient` with path `./data/chroma_db`
- Create collection with appropriate metadata schema
- Store embeddings with complete metadata for citation (FR-ASA-27)
- Implement idempotent operations using hadith IDs

**Schema Design:**

```python
{
    "id": "bukhari_1_1",  # Format: {collection}_{bookId}_{hadithId}
    "embedding": [...],    # 1024-dim vector from multilingual-e5-large
    "document": "...",     # Cleaned Arabic text
    "metadata": {
        "collection": "Sahih al-Bukhari",
        "book_id": 1,
        "chapter_id": 1,
        "chapter_arabic": "كتاب بدء الوحى",
        "chapter_english": "Revelation",
        "hadith_number": 1,
        "narrator_arabic": "...",
        "narrator_english": "...",
        "language": "arabic"
    }
}
```

**Error Handling:**

- Log failed embeddings to `./logs/ingestion_errors.log`
- Continue processing on individual failures
- Final summary report (success/fail counts)

#### 3. Configuration Management (`src/config/settings.py`)

**Status:** Not Started (if empty)

Store all configuration in a central location:

- Modal endpoint URLs
- ChromaDB path
- Batch sizes
- Retry settings
- Logging configuration

Use `python-dotenv` to load from `.env` file for sensitive data.

#### 4. Create Executable Script

Add `if __name__ == "__main__":` block to `src/data/ingestion.py` with:

- Command-line arguments (optional: `--batch-size`, `--dry-run`)
- Progress bar using `tqdm` (add to requirements if needed)
- Summary statistics on completion

---

## Phase 3-6: Agentic Workflow (Next Phase)

### Phase 3: Query Analysis Agent

- Implement intent classification
- Add typo correction for Arabic
- Query decomposition for multi-part questions

### Phase 4: Retrieval Agent

- Semantic search (vector similarity)
- Keyword search (BM25)
- Metadata filtering
- Hybrid search fusion

### Phase 5: Evaluation Agent

- Quality assessment tool
- Grounding validation
- Stopping condition logic

### Phase 6: Synthesis Agent

- Information combination
- Citation generation
- Grounding enforcement
- No-answer detection

---

## Phase 7: Integration & Orchestration (Final Phase)

### Supervisor Agent

- Central workflow orchestrator
- Iteration control
- State management with LangGraph

### FastAPI Application

- REST API endpoints
- Request/response handling
- Integration with all agents

### Performance Requirements

- **NFR-10:** Cold start < 90 seconds
- **NFR-11:** Warm start < 15 seconds

---

## Current Focus: Data Foundation

**Next Actions:**

1. Implement `src/utils/arabic_processing.py`
2. Create `src/data/ingestion.py`
3. Test ingestion with sample data
4. Verify ChromaDB storage and retrieval
5. Run full ingestion on 15,000 hadiths

**Success Criteria:**

- All 7,277 Bukhari hadiths successfully embedded and stored
- Metadata integrity verified (NFR-05)
- Query retrieval test passes
- Documentation complete

---

## Notes

### Embedding Model

- **Model:** `intfloat/multilingual-e5-large`
- **Dimensions:** 768 (verify with Modal response)
- **Optimization:** Arabic-optimized, supports multilingual queries

### Design Decisions

- **Idempotency:** Use hadith IDs as unique identifiers to prevent duplicates
- **Batching:** Process in batches to manage memory and API rate limits
- **Storage:** Separate collections for Arabic and English if needed (TBD)
- **Metadata:** Rich metadata for precise filtering and citation

### Open Questions

1. Should we store English translations in the same collection or separate?
2. Do we need to chunk long hadiths, or is each hadith a single document?
3. What batch size provides optimal performance for Modal endpoint?

---

**Last Updated:** November 17, 2025
