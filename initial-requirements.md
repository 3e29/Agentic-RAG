# Top 30 Core System Requirements

This file outlines the 30 most critical functional (FR) and non-functional (NFR) requirements for the Hadith RAG system, organized by development phase. [cite_start]All requirements are sourced from the "Copy of Graduation Project" document, section 3.3.1.

---

## Phase 1: Data Prep & Infrastructure

This phase covers the foundational setup of the knowledge base and data storage.

* [cite_start]**FR-DIC-33 (Text Preprocessing):** The ingestion tool shall normalize Arabic text, handle diacritics, remove artifacts, and standardize formatting.
* [cite_start]**FR-DIC-35 (Vector Storage):** The pipeline shall persist embeddings and associated metadata in a vector database optimized for semantic search.
* [cite_start]**NFR-05 (Data Consistency):** The system must ensure 100% consistency between retrieved hadith texts and their source metadata (book, chapter, etc.).

---

## Phase 2: Data Ingestion Component

This phase covers the process of transforming the raw JSON data into a searchable vector index.

* [cite_start]**FR-DIC-31 (Data Ingestion Pipeline):** The system shall implement a pipeline to load hadith datasets from JSON, extract metadata, and generate embeddings.
* [cite_start]**FR-DIC-32 (JSON Parsing):** The pipeline shall include a tool to extract hadith text, narrator chains, book references, and hadith numbers from the JSON datasets.
* [cite_start]**FR-DIC-34 (Embedding Generation):** The pipeline shall convert preprocessed hadith chunks into vector representations using Arabic-optimized embedding models.

---

## Phase 3: Query Analysis Agent

This agent is responsible for understanding and deconstructing the user's raw input.

* [cite_start]**FR-QAA-04 (Query Analysis Agent):** The system shall implement this agent to decompose and understand the user's raw input, transforming it into a structured plan.
* [cite_start]**FR-QAA-07 (Intent Classification Tool):** The agent must categorize queries into types, such as thematic search, specific lookup, or comparative analysis.
* [cite_start]**FR-QAA-05 (Typo Correction Tool):** The agent shall include a tool to detect and correct common Arabic spelling errors and diacritic mistakes.
* [cite_start]**FR-QAA-08 (Query Decomposition Tool):** The agent shall include a tool to split multi-part questions into separate sub-queries for processing.

---

## Phase 4: Retrieval Agent

This agent is responsible for finding relevant information in the vector database.

* [cite_start]**FR-RA-11 (Retrieval Planning Agent):** The system shall implement this agent to execute search strategies against the knowledge base based on the query plan.
* [cite_start]**FR-RA-12 (Semantic Search Tool):** The agent must have a tool for vector similarity search (e.g., cosine similarity).
* [cite_start]**FR-RA-13 (Keyword Search Tool):** The agent shall include a lexical matching tool (e.g., BM25) for finding exact terms.
* [cite_start]**FR-RA-14 (Metadata Filter Tool):** The agent must have a tool to filter results based on metadata like book names, chapter titles, or narrator names.
* [cite_start]**FR-RA-15 (Hybrid Search Tool):** The agent shall include a tool to combine and fuse results from semantic and keyword search.

---

## Phase 5: Evaluation Agent

This agent acts as a quality control guardrail, assessing the retrieved information.

* [cite_start]**FR-EA-19 (Evaluation Agent):** The system shall implement this agent to assess the quality, relevance, and faithfulness of retrieved information.
* [cite_start]**FR-EA-20 (Quality Assessment Tool):** The agent must have a tool to analyze the semantic relevance and completeness of retrieved results against the query.
* [cite_start]**FR-EA-23 (Stopping Condition Tool):** The agent must include a tool to autonomously decide when sufficient information has been gathered.
* [cite_start]**FR-EA-24 (Grounding Validation Tool):** The agent must have a tool to verify that all information exists verbatim in the retrieved sources, rejecting non-traceable content.

---

## Phase 6: Answer Synthesis Agent

This agent is responsible for generating the final, human-readable answer.

* [cite_start]**FR-ASA-25 (Answer Synthesis Agent):** The system shall implement this agent to generate a final, coherent, and grounded answer for the user.
* [cite_start]**FR-ASA-26 (Information Combination Tool):** The agent must have a tool to synthesize content from multiple hadiths into a unified narrative.
* [cite_start]**FR-ASA-27 (Citation Generation Tool):** The agent shall include a tool to produce complete references for all cited hadiths (collection, book, number, etc.).
* [cite_start]**FR-ASA-28 (Grounding Enforcement Tool):** The agent must have a tool to ensure every claim in the answer is traceable to a retrieved text, preventing hallucination.
* [cite_start]**FR-ASA-30 (No-Answer Detection Tool):** The agent must identify when retrieved info is insufficient and return an explicit "cannot answer" response.

---

## Phase 7: Supervisor Agent & Integration

This phase covers the orchestration layer and key infrastructure integrations.

* [cite_start]**FR-SA-01 (Supervisor Agent):** The system shall implement a Supervisor Agent as the central orchestrator to manage the workflow and agent state.
* [cite_start]**FR-SA-03 (Iteration Control Tool):** The Supervisor must have a tool to track retrieval cycle counts and enforce maximum iteration limits.
* [cite_start]**FR-FI-36 (External AI Service Integration):** The system shall integrate with externally-hosted, serverless AI endpoints on Modal.com for GPU tasks.
* [cite_start]**FR-FI-37 (LangGraph Workflow Engine):** The system shall use LangGraph as the core engine for stateful, cyclic, multi-agent graphs.
* [cite_start]**NFR-10 (Cold Start Response Time):** On the first request to an idle endpoint, the system shall return a response within 90 seconds.
* [cite_start]**NFR-11 (Warm Start Response Time):** For subsequent requests, the system shall return a response to a standard query in under 15 seconds.