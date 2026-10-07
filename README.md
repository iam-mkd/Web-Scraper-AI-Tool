# 🌐 Autonomous Web Scraper & AI Agent Tool

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-f55036.svg)](https://groq.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-darkgreen.svg)](https://www.langchain.com/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-blueviolet.svg)](https://smith.langchain.com/)

An autonomous AI Agent and Web Scraper API that transforms high-level user tasks (e.g. *"Find me the best DDR5 RAM deals on Amazon under ₹50,000"*) into automated web exploration, anti-bot bypass scraping, deterministic filtering, and intelligent LLM-synthesized insights.

Instead of requiring users to supply static URLs or feeding raw HTML blobs into language models, this system:
1. **Empowers an AI Agent** to determine what tools it needs (`search_amazon`, `scrape_url`, `search_rag`, `extract_information`).
2. **Targets Amazon India (`amazon.in`)** with TLS browser impersonation to bypass anti-bot shields.
3. **Extracts clean structured product records** validated against a typed **`Product` Pydantic model** (`name`, `price`, `capacity_gb`, `speed_mhz`, `cl_latency`, `rating`, `review_count`, `form_factor`, `kit_size`, `value_score`, `url`).
4. **Distinguishes RAM Form Factors**: Clearly classifies and filters Desktop (`UDIMM`) vs Laptop (`SODIMM`) memory kits so users never receive incompatible recommendations.
5. **Applies a Deterministic Value Score Engine**:
   $$\text{Value Score} = \text{Performance (MHz \& CL)} + \text{Capacity} + \text{Rating} + \text{Review Confidence} - \text{Price Penalty (Price/GB)}$$
   Prevents an 8GB stick from being falsely branded as the "best deal" simply because it has the lowest absolute price.
6. **Generates natural language answers** using ultra-fast **Groq LPU Inference** (`openai/gpt-oss-120b`).
7. **Maintains decoupled Ingestion & QA RAG pipelines** backed by **ChromaDB** and **Ollama `nomic-embed-text`**.

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
  - [1. Autonomous Research Task (`POST /api/research`)](#1-autonomous-research-task-post-apiresearch)
  - [2. General Agent Task (`POST /api/agent/task`)](#2-general-agent-task-post-apiagenttask)
  - [3. Scrape & Ingestion Pipeline (`POST /api/scrape`)](#3-scrape--ingestion-pipeline-post-apiscrape)
  - [4. Question Answering RAG Pipeline (`POST /api/query`)](#4-question-answering-rag-pipeline-post-apiquery)
  - [5. Health Status (`GET /health`)](#5-health-status-get-health)
- [Anti-Bot Protection & Edge Cases](#-anti-bot-protection--edge-cases)
- [Running Automated Tests](#-running-automated-tests)
- [Project Directory Structure](#-project-directory-structure)
- [License](#-license)

---

## 🚀 Key Features

- **💻 Interactive Web UI Frontend (`/`)**: Modern, zero-framework Vanilla HTML5/CSS3/ES6 interface with dark glassmorphism styling, live backend health telemetry, quick query preset chips, and dual modes (Autonomous Agent & RAG QA).
- **📡 Real-Time Tool Execution Trace (SSE Streaming `POST /api/agent/stream`)**: True Server-Sent Events (SSE) streaming visualizes every tool invocation as it happens (search -> ranking -> deep inspection -> compare -> synthesis) with live execution timers, pulsing animations, and collapsible `<details>` to inspect exact input parameters and JSON outputs.
- **🛍️ Amazon India Structured Scraper (`structured_scraper.py`)**: Converts Amazon search result and product detail pages into typed `Product` and `ProductDetail` models with prices (INR), memory sizes, clock speeds, CAS latency, ratings, review counts, form factors, kit sizes, operating voltage, warranty, and canonical URLs.
- **🔬 Deep Hardware Spec Inspection (`get_product_details`)**: Inspects individual product pages to extract granular technical tables (Brand, Model, Memory Speed, Operating Voltage, Form Factor, Pin Count, Item Model Number, Country of Origin, Warranty) and feature highlights.
- **⚖️ Deterministic Product Comparison (`compare_products`)**: Side-by-side spec comparison matrix with pros/cons calculation, price-per-GB, and categorical verdicts (`best_value`, `best_performance`, `budget_pick`).
- **📐 Typed Pydantic Schemas (`models.py`)**: Enforces strict `Product` and `ProductDetail` schemas and discards non-RAM accessories (thermal pads, cables).
- **🧮 Deterministic Value Score Engine**: Mathematically scores hardware value based on true hardware parameters (capacity, clock frequency, CAS latency, review confidence, and INR-per-GB) without LLM math hallucination.
- **🖥️ Desktop vs. Laptop RAM Separation**: Identifies form factors (`UDIMM` vs. `SODIMM`) and kit configurations (`1x8GB`, `2x16GB`, etc.).
- **🛡️ Anti-Bot Bypass (`scraper.py`)**: Uses `curl-cffi` to mimic authentic browser TLS/JA3 handshakes and HTTP/2 framing with automatic retry backoff.
- **⚡ Ultra-Fast Inference via Groq**: Powered by `openai/gpt-oss-120b` (120B parameter model) on Groq LPUs for sub-second tool planning and synthesis.
- **💾 Local Vector RAG Pipeline**: Decoupled ingestion and retrieval using Ollama `nomic-embed-text` (768-dim embeddings) and persistent `ChromaDB`.
- **🔍 Full LangSmith Traceability**: Native tracking for every retrieval, prompt assembly, latency metric, and token count.
- **📦 Strict JSON REST API**: Built on `FastAPI` with comprehensive Pydantic validation and strict JSON error handlers.

---

## 🏗️ System Architecture

```
                       ┌────────────────────────────────────────┐
                       │        Client (cURL / Frontend)        │
                       └───────────────────┬────────────────────┘
                                           │ JSON Request
                                           ▼
                       ┌────────────────────────────────────────┐
                       │        FastAPI Server (main.py)        │
                       └─────┬──────────────┬───────────┬──────┬┘
                             │              │           │      │
      POST /api/research     │ POST /agent  │ POST      │ POST │
                             ▼              ▼ /scrape   ▼ /query
       ┌────────────────────────────────────────────────────────┐ ┌─────────┐ ┌─────────┐
       │         Agent Service (agent_service.py)               │ │Ingestion│ │QA (RAG) │
       │  Autonomous Multi-Step ReAct Loop (max 6 steps)        │ │Pipeline │ │Pipeline │
       │  Groq GPT-OSS 120B Agent Brain                         │ └────┬────┘ └────┬────┘
       └──────────────────────────┬─────────────────────────────┘      │           │
                                  │                                    │           │
       Step 1: search_amazon()    │                                    ▼           ▼
       Step 2: get_product_details()                               ┌─────────────────────┐
       Step 3: compare_products() │                                │ Vector Store Layer  │
                                  ▼                                │ • ChromaDB vectors  │
       ┌────────────────────────────────────────────────────────┐  │ • nomic-embed-text  │
       │ Structured Tools & Scrapers (structured_scraper.py)    │  └─────────────────────┘
       │ ├── search_amazon: Scrapes candidates + Value Scores   │
       │ ├── get_product_details: Granular voltage & warranty   │
       │ ├── compare_products: Deterministic head-to-head       │
       │ └── Session Cache: Stores discovered product records   │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │ Final Synthesis (Groq GPT-OSS 120B)                    │
       │ • Crowns Best Value winner using Value Score (0 - 100) │
       │ • Generates side-by-side Markdown comparison table     │
       │ • Clarifies UDIMM (Desktop) vs SODIMM (Laptop)         │
       │ • Technical justification (voltage, latency, warranty) │
       └────────────────────────────────────────────────────────┘
```


---

## 💻 Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | [Python 3.13+](https://www.python.org/) | Core programming language |
| **Agent & Tools** | [LangChain](https://www.langchain.com/) (`langchain-groq`, `langchain-core`) | Tool binding, message chaining, agent orchestration |
| **LLM Inference** | [Groq](https://groq.com/) (`openai/gpt-oss-120b`) | Ultra-fast cloud inference for tool calling and reasoning |
| **Embeddings** | [Ollama](https://ollama.com/) (`nomic-embed-text`) | 768-dimensional dense vector embeddings |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (`langchain-chroma`) | Persistent vector database (`./chroma_db`) |
| **Scraper** | [curl-cffi](https://github.com/yifeikong/curl_cffi) & [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) | Browser TLS fingerprint impersonation & structured parsing |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/) | High-performance asynchronous REST API |
| **Validation** | [Pydantic v2](https://docs.pydantic.dev/) | Strict request/response validation & `Product` data model |
| **Observability** | [LangSmith](https://smith.langchain.com/) | Real-time tracing and latency monitoring |
| **Package Manager**| [uv](https://github.com/astral-sh/uv) | Modern, high-speed Python package manager |

---

## 📋 Prerequisites

1. **Python 3.13+** installed.
2. **[Groq API Key](https://console.groq.com/)** added to `.env` (`GROQ_API_KEY="gsk_..."`).
3. **[Ollama](https://ollama.com/)** running locally for vector embeddings:
   ```bash
   ollama pull nomic-embed-text
   ```
4. A **[LangSmith](https://smith.langchain.com/)** API key (optional, for tracing).

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
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r pyproject.toml
```

---

## 🔑 Environment Configuration

Copy `.env.example` to create your local `.env`:

```bash
cp .env.example .env
```

Configure your `.env` variables:

```env
# Groq LLM API Configuration
GROQ_API_KEY="gsk_your_groq_api_key_here"
GROQ_MODEL="openai/gpt-oss-120b"

# Ollama Endpoint (Used for vector embeddings)
OLLAMA_BASE_URL="http://localhost:11434"

# LangSmith Tracing (Optional)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="lsv2_pt_your_api_key_here"
LANGCHAIN_PROJECT="GenAI_WebScraper_Ollama"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

---

## 🚦 Running the Server

Start the FastAPI application with `uvicorn`:

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Docs**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Reference

All requests and responses strictly accept and return `application/json`.

### 1. Autonomous Research Task (`POST /api/research`)
Dedicated endpoint for structured shopping and product research. Takes a natural language task, determines what to do via GPT-OSS 120B, invokes `search_amazon`, extracts typed `Product` models, evaluates deterministic Value Scores, and synthesizes a grounded `final_answer`.

- **Request Body**:
  ```json
  {
    "task": "Find me the best DDR5 RAM deals on Amazon under ₹50,000"
  }
  ```
- **cURL Example**:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/research \
    -H "Content-Type: application/json" \
    -d '{"task": "Find me the best DDR5 RAM deals on Amazon under ₹50,000"}'
  ```
- **Response (200 OK)**:
  ```json
  {
    "task": "Find me the best DDR5 RAM deals on Amazon under ₹50,000",
    "tool_called": "search_amazon",
    "tool_arguments": {
      "query": "DDR5 RAM",
      "max_price": 50000,
      "form_factor": "UDIMM",
      "optimization": "best_value"
    },
    "stats": {
      "total_scraped": 22,
      "matching_criteria": 15,
      "best_value_product": {
        "name": "Patriot Memory Viper Venom DDR5 RAM 32GB (2X16GB) 6000MT/s CL30 UDIMM...",
        "value_score": 87.0,
        "capacity_gb": 32,
        "speed_mhz": 6000,
        "cl_latency": 30,
        "form_factor": "UDIMM",
        "price": 49199.0,
        "price_per_gb": 1537.5,
        "url": "https://www.amazon.in/dp/B0D4NLSP87"
      },
      "cheapest_product": {
        "name": "CORSAIR Vengeance DDR5 RAM 8GB (1x8GB) 5200MHz CL40...",
        "price": 15279.0,
        "capacity_gb": 8,
        "form_factor": "UDIMM",
        "url": "https://www.amazon.in/dp/B0G61JZHBX"
      },
      "highest_performance_product": {
        "name": "Patriot Memory Viper Venom DDR5 RAM 32GB (2X16GB) 6000MT/s CL30...",
        "speed_mhz": 6000,
        "cl_latency": 30,
        "price": 49199.0
      },
      "highest_rated_product": {
        "name": "EVM 16GB DDR5 5600MHz UDIMM Desktop RAM...",
        "rating": 4.9,
        "review_count": 8,
        "price": 32999.0
      },
      "form_factor_breakdown": {
        "UDIMM_desktop": 15,
        "SODIMM_laptop": 0
      },
      "price_range": {
        "min_price": 15279.0,
        "max_price": 49199.0,
        "avg_price": 32595.93
      }
    },
    "products": [
      {
        "name": "Patriot Memory Viper Venom DDR5 RAM 32GB (2X16GB) 6000MT/s CL30...",
        "price": 49199.0,
        "capacity_gb": 32,
        "speed_mhz": 6000,
        "cl_latency": 30,
        "rating": 4.5,
        "review_count": 658,
        "form_factor": "UDIMM",
        "kit_size": "2x16GB",
        "value_score": 87.0,
        "url": "https://www.amazon.in/dp/B0D4NLSP87"
      }
    ],
    "final_answer": "### Best Value Recommendation\n\n**Patriot Memory Viper Venom DDR5 32GB (2x16GB) 6000MT/s CL30** is the #1 Best Value deal with a Value Score of **87.0/100**.\n\nWhile the Corsair 8GB stick is the cheapest in absolute terms (₹15,279), its price-per-GB is ₹1,910/GB with only 8GB capacity. The Patriot 32GB kit offers true high-end specs (6000MHz, CL30) at a vastly superior ₹1,538/GB.",
    "model": "openai/gpt-oss-120b"
  }
  ```

---

### 2. General Agent Task (`POST /api/agent/task`)
Accepts a natural language task across all registered tools (`search_amazon`, `search_web`, `scrape_url`, `search_rag`, `extract_information`).

- **Request Body**:
  ```json
  {
    "task": "Find me DDR5 RAM deals on Amazon under 50000"
  }
  ```
- **Response (200 OK)**:
  Returns `task`, `tool_called`, `tool_arguments`, `stats`, `products`, and `explanation`.

---

### 3. Scrape & Ingestion Pipeline (`POST /api/scrape`)
Executes the isolated Ingestion pipeline: fetches a webpage, cleans HTML, splits into chunks, embeds with `nomic-embed-text`, and stores in ChromaDB.

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
    "char_count": 156,
    "message": "Successfully scraped, chunked, and stored 1 chunks in ChromaDB."
  }
  ```

---

### 4. Question Answering RAG Pipeline (`POST /api/query`)
Executes the isolated QA pipeline against already indexed ChromaDB vectors.

- **Request Body**:
  ```json
  {
    "url": "https://example.com",
    "query": "What is the purpose of this domain?",
    "auto_scrape": true
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "answer": "This domain is established to illustrate examples in documentation without requiring coordination or permission.",
    "sources": [
      {
        "content": "This domain is for use in documentation examples...",
        "metadata": {
          "url": "https://example.com",
          "chunk_id": 0,
          "title": "Example Domain"
        },
        "score": 0.4978
      }
    ],
    "url": "https://example.com",
    "query": "What is the purpose of this domain?",
    "model": "openai/gpt-oss-120b"
  }
  ```

---

### 5. Health Status (`GET /health`)
Checks connectivity to Groq, Ollama, and LangSmith.

- **Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "groq_service": "connected",
    "ollama_service": "connected",
    "embedding_model": "nomic-embed-text",
    "llm_model": "openai/gpt-oss-120b",
    "langsmith_tracing": true,
    "langsmith_project": "GenAI_WebScraper_Ollama"
  }
  ```

---

## 🛡️ Anti-Bot Protection & Edge Cases

- **Amazon AWS WAF & Cloudflare**: Standard HTTP clients (Requests, httpx, urllib) fail with `HTTP 503 Service Unavailable` due to mismatched OpenSSL TLS/JA3 fingerprints. We use `curl-cffi` with authentic Chrome TLS impersonation and automatic exponential retry backoff.
- **Login-Walled Sites (LinkedIn)**: URLs behind mandatory user authentication return `HTTP 999 Request Denied`. The scraper catches this and raises a clean `403 Forbidden` response explaining the authentication wall.

---

## 🧪 Running Automated Tests

Run the comprehensive test suites:

### 1. Test Structured Research & Value Score Engine (`test_research.py`):
```bash
uv run python test_research.py
```
*Verifies typed `Product` model validation, `form_factor` (UDIMM vs SODIMM), `kit_size`, deterministic `value_score`, spec-aware filtering, and the `POST /api/research` endpoint.*

### 2. Test Agent Task Workflow (`test_agent.py`):
```bash
uv run python test_agent.py
```
*Verifies multi-turn tool calling, live Amazon scraping, and `POST /api/agent/task`.*

### 3. Test Decoupled Ingestion & QA Pipelines (`test_app.py`):
```bash
uv run python test_app.py
```
*Verifies HTML scraping, chunking, embedding in ChromaDB, RAG retrieval, and FastAPI endpoints.*

---

## 📁 Project Directory Structure

```text
web_scraper_ai/
├── .env.example          # Environment variable template
├── .gitignore            # Git exclusion rules
├── pyproject.toml        # Dependencies and project specifications
├── README.md             # Comprehensive project documentation
├── models.py             # Pydantic schemas (Product, ResearchRequest, ResearchResponse)
├── scraper.py            # Anti-bot web scraper with TLS impersonation
├── structured_scraper.py # Structured Amazon product parser with retry backoff
├── agent_service.py      # Autonomous AI Agent Brain with Value Score engine
├── ingestion.py          # Decoupled Ingestion Pipeline (fetch -> chunk -> embed -> ChromaDB)
├── qa.py                 # Decoupled QA Pipeline (query -> ChromaDB -> Groq LLM)
├── vector_store.py       # ChromaDB vector store manager & Ollama embeddings
├── main.py               # FastAPI application exposing /api/research & RAG endpoints
├── test_research.py      # Automated tests for Value Score & /api/research
├── test_agent.py         # End-to-end integration tests for the AI Agent
└── test_app.py           # Integration tests for Ingestion and QA pipelines
```

---

## 📄 License

Distributed under the [MIT License](LICENSE). Feel free to use, modify, and distribute for personal and commercial projects.
