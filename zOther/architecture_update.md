# Project Architecture Update: Decoupling AI Model Service

## 1. Executive Summary

This document outlines a major architectural shift for the Agentic RAG system. To solve performance, scalability, and cost issues related to the 14B+ LLM (Qwen2.5), we are moving from a single-server (monolithic) design to a **decoupled, microservice-based architecture**.

## 2. The New Architecture: "Brain" and "Muscle"

We are splitting the system into two core components:

### 🧠 The "Brain" (Main Application)

* **Technology:** FastAPI, LangGraph, Uvicorn, Gunicorn
* **Role:** This is the lightweight, main application server. It handles all user HTTP requests, runs the **Supervisor Agent**, manages `AgentState`, and orchestrates the entire agentic workflow.
* **Host:** It can run on any cheap, low-CPU server (e.g., local machine, free-tier VM, Railway, Heroku) because it does no heavy computation.

### 🦾 The "Muscle" (AI Model Service)

* **Technology:** **Modal.com**
* **Role:** This is a serverless, GPU-powered service that does *only* the heavy lifting. It exposes two endpoints:
    1.  The Qwen2.5 14B model for reasoning and synthesis.
    2.  The `multilingual-e5-large` model for embedding.
* **Host:** Runs on-demand on Modal's high-performance GPUs (e.g., A10G).

## 3. Key Benefits of This Change

1.  **Cost-Efficiency:** We now pay for expensive GPU time *per second* of use, not for a 24/7 dedicated server.
2.  **Scalability:** Modal can automatically scale to handle hundreds of concurrent users, whereas our local server cannot.
3.  **No Local Hardware:** The project can be developed and run by anyone, anywhere, without needing a personal NVIDIA GPU.
4.  **Performance:** We solve the 10GB model download problem by using a **`modal.Volume`**. This is a one-time download, creating a persistent cache. This makes our cold starts (loading the model from cache to GPU) fast and efficient.