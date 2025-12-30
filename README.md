# Project Overview: Hadith RAG System

This project is a Retrieval-Augmented Generation (RAG) system designed to answer user questions based on a private collection of 15,000 Hadiths.

The primary goal is to provide accurate, reliable, and contextually-aware responses by grounding a large language model (LLM) in this specific knowledge base.

---

## Architecture & Approach

The system uses a decoupled "Brain" (orchestrator) and "Muscle" (models) architecture.

### 1. Core Technologies

* **Orchestrator (The Brain):** A **FastAPI** application that runs locally. It manages the user request, orchestrates the agentic workflow, and serves the final API.
* **Vector Database (The Memory):** **ChromaDB** (using `PersistentClient`). This database is stored in a local directory (`./hadith_db`) and contains the 15,000 Hadith vectors.
* **Models (The Muscle):** Hosted on **Modal** for scalable, serverless GPU inference. We interact with two key endpoints:
    * **LLM Endpoint (`/generate`):** Uses `qwen2.5-14b-instruct` for all reasoning, evaluation, and text generation tasks.
    * **Embedding Endpoint (`/embed`):** Uses `intfloat/multilingual-e5-large` for vectorizing user queries and (one-time) a-priori document ingestion.

---

### 2. Agentic RAG Workflow

Instead of a simple "retrieve-then-answer" pipeline, this project uses a multi-agent system orchestrated by FastAPI for more robust and accurate responses.

1.  **Query Analysis:**
    * An `QueryAnalysisAgent` (using the Modal `/generate` LLM) receives the user's raw query.
    * It refines, rewrites, and expands the query to be optimal for vector search (e.g., adds keywords, clarifies intent).

2.  **Retrieval:**
    * A `RetrievalAgent` takes the *new* refined query.
    * It calls the Modal `/embed` endpoint to turn the query into a vector.
    * It uses this vector to search the **ChromaDB** collection and find the top-k relevant Hadith texts.

3.  **Evaluation:**
    * An `EvaluationAgent` (using the Modal `/generate` LLM) receives the retrieved Hadiths and the original query.
    * It acts as a "guardrail," judging if the retrieved documents are relevant *at all* and if they are sufficient to answer the question.

4.  **Synthesis:**
    * If the documents are deemed relevant, a `SynthesisAgent` (using the Modal `/generate` LLM) receives the original query and the relevant Hadith snippets.
    * It is prompted to draft a final, comprehensive, and well-cited answer based *only* on the provided context.

5.  **Supervisor:**
    * A `SupervisorAgent` (or the main FastAPI router) oversees this process, deciding which agent to call next and handling the final response generation back to the user.