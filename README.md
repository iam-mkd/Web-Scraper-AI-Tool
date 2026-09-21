# 🌐 Web Scraper AI Tool

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-black.svg)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-darkgreen.svg)](https://www.langchain.com/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-blueviolet.svg)](https://smith.langchain.com/)

A modern, privacy-focused, AI-powered Web Scraper and Question-Answering REST API.

Given any webpage URL, it scrapes and sanitizes HTML content, bypasses modern anti-bot shields (like Amazon AWS WAF / Cloudflare), splits text into semantic chunks, generates vector embeddings using **Ollama `nomic-embed-text`**, persists them in **ChromaDB**, and answers user questions through grounded RAG generation using **Ollama `granite3-dense:2b`**.

All execution steps, prompt constructions, and LLM inferences are fully traced via **LangSmith**, and exposed via a **strict JSON-only FastAPI REST API**.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Environment Configuration](#-environment-configuration)
- [Running the Server](#-running-the-server)
- [API Reference](#-api-reference)
  - [1. Health Check](#1-health-check)
  - [2. Query Webpage (Auto-Scrape + RAG)](#2-query-webpage-auto-scrape--rag)
  - [3. Explicit Scrape & Index](#3-explicit-scrape--index)
- [Anti-Bot Protection & Edge Cases](#-anti-bot-protection--edge-cases)
- [Running Automated Tests](#-running-automated-tests)
- [Project Directory Structure](#-project-directory-structure)
- [License](#-license)

---

## 🚀 Key Features

- **🛡️ Anti-Bot Bypass Scraping**: Employs `curl-cffi` to simulate authentic browser TLS/JA3 fingerprints and HTTP/2 framing, seamlessly accessing anti-bot protected sites (e.g. Amazon search and product listings).
- **🧹 Intelligent HTML Cleaning**: Strips away irrelevant clutter (`<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`, forms, SVGs) to preserve clean, high-density text.
- **✂️ Semantic Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter` with intelligent overlapping to preserve contextual continuity.
- **⚡ 100% Local Inference**:
  - **Embeddings**: `nomic-embed-text` (768-dimensional dense vectors) via Ollama.
  - **Generative LLM**: `granite3-dense:2b` (fast, concise, private) via Ollama.
- **💾 Persistent Vector DB**: Stores documents, chunk indices, and metadata in `ChromaDB` (`./chroma_db`) with URL-scoped querying.
- **🔍 Full LangSmith Traceability**: Native tracking for every retrieval, prompt assembly, latency metric, and token count.
- **📦 Strict JSON REST API**: Built on `FastAPI` with comprehensive Pydantic validation and strict JSON error handlers.

---

## 🏗️ System Architecture

```
                      ┌──────────────────────────────┐
                      │    Client (cURL / Web App)   │
                      └──────────────┬───────────────┘
                                     │ JSON Request
                                     ▼
                      ┌──────────────────────────────┐
                      │    FastAPI Server (main.py)  │
                      └──────┬────────────────┬──────┘
        POST /api/scrape     │                │     POST /api/query
                             ▼                ▼
     ┌────────────────────────────────┐   ┌────────────────────────────────┐
     │      scraper.py (Fetcher)      │   │     rag_service.py (RAG)       │
     │ ────────────────────────────── │   │ ────────────────────────────── │
     │ • curl-cffi (Browser Imperson) │   │ • RecursiveTextSplitter        │
     │ • BeautifulSoup4 HTML Parsing  │   │ • Ollama nomic-embed-text      │
     │ • Whitespace & Noise Stripping │   │ • ChromaDB Persistent Store    │
     └────────────────────────────────┘   │ • Ollama granite3-dense:2b     │
                                          └──────────────┬─────────────────┘
                                                         │
                                                         ▼
                                          ┌────────────────────────────────┐
                                          │     LangSmith Traceability     │
                                          │   (Prompts, Latency, Metrics)  │
                                          └────────────────────────────────┘
```

---

## 💻 Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | [Python 3.13+](https://www.python.org/) | Core programming language |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance async REST API framework |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) | Lightning-fast ASGI production web server |
| **Scraper & Parser** | [curl-cffi](https://github.com/yifeikong/curl_cffi) & [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) | Anti-bot TLS fingerprint impersonation & HTML cleaning |
| **Embedding Model** | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) | 768-dim embedding model running locally via Ollama |
| **LLM** | [granite3-dense:2b](https://ollama.com/library/granite3-dense) | High-performance 2.6B parameter model via Ollama |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Local persistent vector store (`./chroma_db`) |
| **Orchestration** | [LangChain](https://www.langchain.com/) (`langchain-ollama`, `langchain-chroma`) | RAG pipeline, document chunking, prompt chaining |
| **Observability** | [LangSmith](https://smith.langchain.com/) | Distributed tracing, monitoring, and prompt debugging |
| **Validation** | [Pydantic v2](https://docs.pydantic.dev/) | Strict JSON request/response schema validation |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) (or pip) | Ultra-fast Python package resolver and runner |

---

## 📋 Prerequisites

1. **Python 3.13+** installed.
2. **[Ollama](https://ollama.com/)** installed and running on `http://localhost:11434`.
3. Pull the required models in Ollama:
   ```bash
   ollama pull nomic-embed-text
   ollama pull granite3-dense:2b
   ```
4. A **[LangSmith](https://smith.langchain.com/)** account and API key (optional, but recommended for full traceability).

---

## ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/web-scraper-ai.git
cd web-scraper-ai
```

### Option A: Using `uv` (Recommended)

```bash
uv sync
```

### Option B: Using standard `pip`

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r pyproject.toml
# Or install dependencies directly:
pip install fastapi uvicorn pydantic beautifulsoup4 httpx curl-cffi chromadb langchain langchain-community langchain-ollama langchain-chroma langsmith python-dotenv
```

---

## 🔑 Environment Configuration

Copy `.env.example` to create your local `.env`:

```bash
cp .env.example .env
```

Configure your `.env` variables:

```env
# LangSmith Tracing
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="lsv2_pt_your_api_key_here"
LANGCHAIN_PROJECT="GenAI_WebScraper_Ollama"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

# Ollama Endpoint (Default)
OLLAMA_BASE_URL="http://localhost:11434"
```

> [!TIP]
> If you don't have a LangSmith key yet, set `LANGCHAIN_TRACING_V2="false"`. The RAG pipeline will continue to work normally without remote tracing.

---

## 🚦 Running the Server

Start the FastAPI application with `uvicorn`:

```bash
# Using uv:
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Or directly with python/uvicorn:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Docs**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Reference

All requests and responses strictly accept and return `application/json`.

### 1. Health Check
Verify that Ollama, ChromaDB, and LangSmith tracing are active.

- **Method**: `GET /health`
- **Request**:
  ```bash
  curl -s http://127.0.0.1:8000/health
  ```
- **Response**:
  ```json
  {
    "status": "healthy",
    "ollama_service": "connected",
    "embedding_model": "nomic-embed-text",
    "llm_model": "granite3-dense:2b",
    "langsmith_tracing": true,
    "langsmith_project": "GenAI_WebScraper_Ollama"
  }
  ```

---

### 2. Query Webpage (Auto-Scrape + RAG)
The primary endpoint. It takes a webpage URL and a query. If the webpage has not been indexed yet, it automatically fetches, cleans, chunks, and stores it in ChromaDB before generating the grounded answer.

- **Method**: `POST /api/query`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "url": "https://www.amazon.in/s?k=ddr+5+ram+32gb",
    "query": "What 32GB DDR5 RAM options are available and what are their speeds?",
    "auto_scrape": true,
    "force_refresh": false
  }
  ```
- **cURL Example**:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/query \
    -H "Content-Type: application/json" \
    -d '{
      "url": "https://www.amazon.in/s?k=ddr+5+ram+32gb",
      "query": "What 32GB DDR5 RAM options are available and what are their speeds?"
    }'
  ```
- **Response (200 OK)**:
  ```json
  {
    "url": "https://www.amazon.in/s?k=ddr+5+ram+32gb",
    "query": "What 32GB DDR5 RAM options are available and what are their speeds?",
    "answer": "Based on the provided context, here are the 32GB DDR5 RAM options available along with their speeds:\n\n1. Patriot Memory Viper Venom DDR5 32GB (2 x 16GB) 6000MHz UDIMM\n2. Corsair Vengeance RGB RS DDR5 32GB (2 x 16GB) Up to 6000MHz\n3. TeamGroup T-Force Delta RGB 2x16GB 6000MHz (6000MT/s) CL30\n4. Patriot Memory Viper Xtreme 5 RGB DDR5 8000MT/s\n\nThe speeds of these options range between 6000MHz and 8000MT/s.",
    "sources": [
      "1-16 of 105 results for \"ddr 5 ram 32gb\" ... Patriot Memory Viper Venom DDR5 32GB 6000MHz ...",
      "Corsair Vengeance RGB RS DDR5 32GB (2 x 16GB) Up to 6000MHz AMD Intel RAM ..."
    ],
    "model": "granite3-dense:2b"
  }
  ```

---

### 3. Explicit Scrape & Index
Scrape and chunk a webpage in advance without querying immediately.

- **Method**: `POST /api/scrape`
- **Request Body**:
  ```json
  {
    "url": "https://example.com",
    "force_refresh": false
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "status": "success",
    "url": "https://example.com",
    "title": "Example Domain",
    "chunks_indexed": 1,
    "message": "Successfully scraped, chunked, and stored 1 chunks in ChromaDB."
  }
  ```

---

## 🛡️ Anti-Bot Protection & Edge Cases

### Amazon / Cloudflare / WAF-Protected Sites
- Traditional Python scrapers receive `HTTP 503 Service Unavailable` or `403 Forbidden` because their OpenSSL TLS fingerprints do not match real browsers.
- This scraper utilizes **`curl-cffi`** with Chrome JA3/TLS handshake impersonation to transparently pass perimeter bot checks on sites like Amazon.

### Login-Walled Platforms (e.g., LinkedIn)
- Pages behind mandatory user authentication (e.g. LinkedIn personal profiles `/in/*`) return proprietary status codes like **`HTTP 999 Request Denied`**.
- The API catches this condition and returns an informative `403 Forbidden` response explaining the authentication requirement rather than failing with an unhandled exception.

---

## 🧪 Running Automated Tests

An automated test suite (`test_app.py`) is included to verify the scraper, vector store, local Ollama generation, and API endpoints:

```bash
uv run python test_app.py
```

Expected output:
```text
=== 1. Testing Scraper ===
Title: Example Domain
Content length: 127 chars

=== 2. Testing RAG Service ===
Indexed chunks: 1

=== 3. Testing RAG Query with Ollama ===
Query: What is the purpose of this domain?
Answer: The purpose of this domain is for use in documentation examples...
Sources retrieved: 1

=== 4. Testing FastAPI Endpoints ===
Health Response (200): ...
Scrape Response (200): ...
Query Response (200): ...

All automated tests PASSED successfully!
```

---

## 📁 Project Directory Structure

```text
web_scraper_ai/
├── .env.example          # Template for environment variables
├── .gitignore            # Git exclusion rules (ignores .env, chroma_db/, .venv)
├── pyproject.toml        # Project dependencies and metadata
├── README.md             # Project documentation
├── scraper.py            # Async web scraper with browser impersonation
├── rag_service.py        # Text chunking, ChromaDB store, Ollama embeddings & RAG chain
├── main.py               # FastAPI application with strict JSON endpoints
└── test_app.py           # End-to-end integration test suite
```

---

## 📄 License

Distributed under the [MIT License](LICENSE). Feel free to use, modify, and distribute for personal and commercial projects.
