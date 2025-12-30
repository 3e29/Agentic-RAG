You are a Senior Python Backend Engineer specializing in RAG systems. We are implementing the **Retrieval Agent** for a high-precision Islamic scholarship system.

**Context:**
We need to build `src/agents/retrieval.py` and its associated tools in `src/tools/retrieval/`. This agent is responsible for fetching authentic Hadiths from Ṣaḥīḥ al-Bukhārī and Ṣaḥīḥ Muslim. It must be a **"Real Agent"** capable of reasoning about *how* to search (e.g., "The user gave a specific hadith number, so I should use metadata filtering, not vector search").

**Critical Requirement:** The agent must handle **atomic sub-queries** in PARALLEL. Each sub-query runs as an independent "worker" task that can self-correct (retry/relax filters) before reporting back.

**The Architecture (Map-Reduce):**
1.  **Map:** If multiple `sub_queries` exist, spawn parallel async tasks.
2.  **Process:** Each task performs independent `MetadataExtraction` -> `HybridSearch`. If no results, it relaxes filters and retries (up to 3 times) *independently* of other tasks.
3.  **Reduce:** Collect all results and pass them to the `ResultAggregationTool` to produce one unified list.

**Technical Constraints & Best Practices:**
* **Architecture:** Use **LangGraph** for the agent node.
* **Data Validation:** Use **Pydantic V2** for all tool inputs/outputs.
* **Dependency Injection:** Pass the vector store client (e.g., ChromaDB or Qdrant) and BM25 retriever into the tools/agent via the state or configuration, not global variables.
* **Error Handling:** "Fail gracefully." If vector search fails, fallback to keyword search automatically.
* **Zero Hallucination:** The agent must only return raw documents; it does *not* generate answers.

### **Task 1: Implement the 7 Core Tools (`src/tools/retrieval/`)**
Use Pydantic V2 for all schemas.
1.  **`SemanticSearchTool` (FR-RA-12)**
    * **Logic:** Embed query -> Vector Search (Cosine Similarity).
    * **Input:** `query` (str), `k` (int).
    * **Output:** List of Documents with similarity scores.

2.  **`KeywordSearchTool` (FR-RA-13)**
    * **Logic:** Lexical search using BM25 (exact term matching).
    * **Input:** `query` (str), `k` (int).

3.  **`MetadataFilterTool` (FR-RA-14)**
    * **Logic:** Converts natural language constraints (e.g., "in Bukhari Book of Faith") into database filters (e.g., `{"collection": "bukhari", "book": "faith"}`).

4.  **`HybridSearchTool` (FR-RA-15)**
    * **Logic:**
        * Run `SemanticSearchTool`.
        * Run `KeywordSearchTool`.
        * **Fusion:** Combine results using Reciprocal Rank Fusion (RRF) algorithm.
    * **Input:** `query` (str), `alpha` (float).

5.  **`ResultAggregationTool`** 
    * **Logic:**
        1.  **Flatten** results from multiple sub-queries.
        2.  **Deduplicate** by `hadith_id`.
        3.  **RERANK:** Apply Cross-Encoder scoring (or weighted boosting) to sort the final list by actual relevance.
    * **Input:** `raw_results` (List[List[Document]]).
    * **Output:** `final_ranked_docs` (List[Document]).

6.  **`UserHadithProcessorTool` (FR-RA-17)**
    * **Logic:** If the user provided text in the query, index it temporarily into a FAISS/In-memory store and search it.

7.  **`QueryExpansionTool`** 
    * **Logic:** Generate 3-5 synonyms/translations (e.g., "Zakat" -> "Charity", "Alms").
    * **Input:** `query` (str).
    * **Output:** `expanded_queries` (List[str]).
--- 

### **Task 2: Implement the Retrieval Agent (`src/agents/retrieval.py`)**

Implement `retrieval_node` using `asyncio` for parallelism.

**The Workflow:**
1.  **Input Analysis:**
    * If source is `user_text`/`file`, route to respective tools.
    * If `base_knowledge`, proceed to search.

2.  **Query Expansion (The "Smart" Step):**
    * Run `QueryExpansionTool` on the main query (or sub-queries) to get better terms.

3.  **Parallel Execution Loop (The "Map"):**
    * Define `_execute_search_strategy(query, expanded_terms)`:
        * **Step A:** Extract Filters (`MetadataFilterTool`).
        * **Step B:** Try `HybridSearchTool` (Strict Filters).
        * **Step C (Self-Correction):**
            * *If 0 results:* Relax filters (remove 'chapter'). Retry.
            * *If still 0:* Relax filters (remove 'book'). Retry.
    * **Run:** Use `asyncio.gather` to run this for every sub-query.

4.  **Aggregation (The "Reduce"):**
    * Pass all results to `ResultAggregationTool` (which handles deduplication and **Reranking**).
    * Update `state["retrieved_docs"]`.

---

### **Task 3: Robust Testing (`tests/test_retrieval.py`)**
Use `pytest-asyncio`.

1.  **Test Expansion:** Verify "Prayer" expands to include "Salah".
2.  **Test Parallel Map-Reduce:**
    * Input: `["Zakat rules", "Fasting rules"]`.
    * Mock: Zakat search finds [Doc A], Fasting search finds [Doc B].
    * Expectation: Aggregation returns [Doc A, Doc B].
3.  **Test Reranking:**
    * Input: [Doc A (Score 0.9), Doc A (Score 0.8), Doc B (Score 0.5)].
    * Expectation: Result is unique [Doc A, Doc B] with Doc A ranked first.
4.  **Test Self-Correction:**
    * Mock: Strict search fails -> Relaxed search succeeds.
    * Expectation: Agent returns results (not empty).

**Deliverables:**
Provide code for `src/tools/retrieval/` (grouping related tools) and the full `retrieval_node` logic.