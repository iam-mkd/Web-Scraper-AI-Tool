# 🌐 Autonomous Web Scraper & AI Agent Tool

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-black.svg)](https://ollama.com/)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-f55036.svg)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-darkgreen.svg)](https://www.langchain.com/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-blueviolet.svg)](https://smith.langchain.com/)
[![Tests](https://img.shields.io/badge/tests-56%2F56%20passed-brightgreen.svg)]()
[![Offline Golden Eval](https://img.shields.io/badge/golden--eval-30%2F30%20(100%25%20offline)-brightgreen.svg)]()
[![Live LLM Benchmark](https://img.shields.io/badge/live--eval-18%2F30%20(60.0%25%20accuracy)-yellow.svg)]()

An autonomous AI Agent and Web Scraper API that transforms high-level user tasks (e.g. *"Find me the best DDR5 RAM deals on Amazon under ₹50,000"*) into automated web exploration, anti-bot bypass scraping, deterministic filtering, and intelligent LLM-synthesized insights.

Instead of requiring users to supply static URLs or feeding raw HTML blobs into language models, this system:
1. **Empowers an AI Agent** to determine what tools it needs (`search_amazon`, `get_product_details`, `compare_products`, `scrape_url`, `search_rag`, `extract_information`).
2. **Targets Amazon India (`amazon.in`)** with TLS browser impersonation to bypass anti-bot shields.
3. **Extracts clean structured product records** validated against a typed **`Product` Pydantic model** (`name`, `price`, `capacity_gb`, `speed_mhz`, `cl_latency`, `rating`, `review_count`, `form_factor`, `kit_size`, `value_score`, `url`).
4. **Distinguishes RAM Form Factors**: Clearly classifies and filters Desktop (`UDIMM`) vs Laptop (`SODIMM`) memory kits so users never receive incompatible recommendations.
5. **Applies a Deterministic Value Score Engine**:
   $$\text{Value Score} = \text{Performance (MHz \& CL)} + \text{Capacity} + \text{Rating} + \text{Review Confidence} - \text{Price Penalty (Price/GB)}$$
   Prevents an 8GB stick from being falsely branded as the "best deal" simply because it has the lowest absolute price.
6. **Flexible Hybrid LLM Inference**:
   - **Local Inference (Default)**: Zero-cost, unlimited local execution via **Ollama** (`qwen3.5:4b`) with native tool-calling support.
   - **Cloud Inference (Fallback)**: High-speed LPUs via **Groq** (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`) or **Google GenAI / Gemini**.
7. **Maintains decoupled Ingestion & QA RAG pipelines** backed by **ChromaDB** and **Google EmbeddingGemma-2 (`google/embeddinggemma-2`)** via **`sentence-transformers`**.
8. **Deterministic Golden Evaluation & Grader Suite**: 30 curated test cases across 5 categories measuring intent extraction, tool selection, product ranking, grounding, and edge-case resilience with 100% offline accuracy.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Environment Configuration](#-environment-configuration)
- [Running the Server](#-running-the-server)
- [Deterministic Golden Evaluation Suite](#-deterministic-golden-evaluation-suite)
  - [Golden Dataset Structure](#golden-dataset-structure-30-cases)
  - [Deterministic Evaluation Graders](#deterministic-evaluation-graders)
  - [Latest Evaluation Status & Report](#latest-evaluation-status--report)
  - [Running the Evaluation Suite](#running-the-evaluation-suite)
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
- **📡 Real-Time Tool Execution Trace (SSE Streaming `POST /api/agent/stream`)**: True Server-Sent Events (SSE) streaming visualizes every tool invocation as it happens (`search_amazon` -> ranking -> deep inspection -> `compare_products` -> synthesis) with live execution timers, pulsing animations, and collapsible `<details>` to inspect exact input parameters and JSON outputs.
- **🦙 Native Local LLM Inferencing (Ollama `qwen3.5:4b`)**: Uses local Ollama instance with native tool calling (`bind_tools`), avoiding cloud API token limits and 429 rate throttles.
- **🛒 Amazon India Structured Scraper (`amazon_scraper.py`)**: Converts Amazon search result and product detail pages into typed `Product` and `ProductDetail` models with prices (INR), memory sizes, clock speeds, CAS latency, ratings, review counts, form factors, kit sizes, operating voltage, warranty, and canonical URLs.
- **🔬 Deep Hardware Spec Inspection (`get_product_details`)**: Inspects individual product pages to extract granular technical tables (Brand, Model, Memory Speed, Operating Voltage, Form Factor, Pin Count, Item Model Number, Country of Origin, Warranty) and feature highlights.
- **⚖️ Deterministic Product Comparison (`compare_products`)**: Side-by-side spec comparison matrix with pros/cons calculation, price-per-GB, and categorical verdicts (`best_value`, `best_performance`, `budget_pick`).
- **📐 Typed Pydantic Schemas (`models.py`)**: Enforces strict `Product` and `ProductDetail` schemas and discards non-RAM accessories (thermal pads, cables).
- **🧮 Deterministic Value Score Engine**: Mathematically scores hardware value based on true hardware parameters (capacity, clock frequency, CAS latency, review confidence, and INR-per-GB) without LLM math hallucination.
- **🖥️ Desktop vs. Laptop RAM Separation**: Identifies form factors (`UDIMM` vs. `SODIMM`) and kit configurations (`1x8GB`, `2x16GB`, etc.).
- **🛡️ Anti-Bot Bypass (`base_scraper.py`)**: Uses `curl-cffi` to mimic authentic browser TLS/JA3 handshakes and HTTP/2 framing with automatic retry backoff.
- **💾 Local Vector RAG Pipeline**: Decoupled ingestion and retrieval using Google EmbeddingGemma-2 (`google/embeddinggemma-2`) via `sentence-transformers` (768-dim embeddings) and persistent `ChromaDB`.
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
                       └─────┬──────────────┬───────────┬──────┬─┘
                             │              │           │      │
      POST /api/research     │ POST /agent  │ POST      │ POST │
                             ▼              ▼ /scrape   ▼ /query
       ┌────────────────────────────────────────────────────────┐ ┌─────────┐ ┌─────────┐
       │         Agent Service (agent/service.py)               │ │Ingestion│ │QA (RAG) │
       │  Autonomous Multi-Step ReAct Loop (max 6 steps)        │ │Pipeline │ │Pipeline │
       │  Ollama qwen3.5:4b / Groq Cloud Fallback               │ └────┬────┘ └────┬────┘
       └──────────────────────────┬─────────────────────────────┘      │           │
                                  │                                    │           │
       Step 1: search_amazon()    │                                    ▼           ▼
       Step 2: get_product_details()                               ┌─────────────────────┐
       Step 3: compare_products() │                                │ Vector Store Layer  │
                                  ▼                                │ • ChromaDB vectors  │
       ┌────────────────────────────────────────────────────────┐  │ • embeddinggemma-2  │
       │ Structured Tools & Scrapers (scrapers/)                │  └─────────────────────┘
       │ ├── search_amazon: Scrapes candidates + Value Scores   │
       │ ├── get_product_details: Granular voltage & warranty   │
       │ ├── compare_products: Deterministic head-to-head       │
       │ └── Session Cache: Stores discovered product records   │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │ Final Synthesis (qwen3.5:4b / gpt-oss-20b)             │
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
| **Agent & Tools** | [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-ollama`, `langchain-groq`) | Tool binding, message chaining, agent orchestration |
| **Local LLM** | [Ollama](https://ollama.com/) (`qwen3.5:4b`) | Zero-cost local LLM inference with native function calling |
| **Cloud LLM** | [Groq](https://groq.com/) (`openai/gpt-oss-20b`) | Ultra-fast cloud fallback inference |
| **Embeddings** | [SentenceTransformers](https://sbert.net/) (`google/embeddinggemma-2`) | 768-dimensional dense vector embeddings with `Document`/`SearchQuery` prompts |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (`langchain-chroma`) | Persistent vector database (`./chroma_db`) |
| **Scraper** | [curl-cffi](https://github.com/yifeikong/curl_cffi) & [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) | Browser TLS fingerprint impersonation & structured parsing |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/) | High-performance asynchronous REST API |
| **Validation** | [Pydantic v2](https://docs.pydantic.dev/) | Strict request/response validation & `Product` data model |
| **Evaluation** | Custom Deterministic Graders | Intent extraction, tool selection, ranking, and grounding benchmarks |
| **Observability** | [LangSmith](https://smith.langchain.com/) | Real-time tracing and latency monitoring |
| **Package Manager**| [uv](https://github.com/astral-sh/uv) | Modern, high-speed Python package manager |

---

## 📋 Prerequisites

1. **Python 3.13+** installed.
2. **[Ollama](https://ollama.com/)** running locally:
   ```bash
   ollama pull qwen3.5:4b
   ```
3. *(Optional)* **[Groq API Key](https://console.groq.com/)** for cloud inference (`GROQ_API_KEY="gsk_..."`).
4. *(Optional)* **[Hugging Face Token](https://huggingface.co/settings/tokens)** for EmbeddingGemma-2 (`HF_TOKEN="hf_..."`).
5. *(Optional)* **[LangSmith](https://smith.langchain.com/)** API key for observability tracing.

---

## ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/web-scraper-ai.git
cd web-scraper-ai
```

### Install Dependencies via `uv`

```bash
uv sync
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
# Local Ollama Configuration (Default)
OLLAMA_MODEL="qwen3.5:4b"
OLLAMA_BASE_URL="http://localhost:11434"

# Cloud Fallback (Optional)
GROQ_API_KEY="gsk_..."
GROQ_MODEL="openai/gpt-oss-20b"
HF_TOKEN="hf_..."

# LangSmith Tracing (Optional)
LANGCHAIN_TRACING_V2="false"
LANGCHAIN_API_KEY="lsv2_..."
LANGCHAIN_PROJECT="GenAI_WebScraper_Ollama"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

---

## 🎯 Deterministic Golden Evaluation Suite

The project includes an **evaluation measurement framework** designed to benchmark LLM agent accuracy without altering application business logic.

### Golden Dataset Structure (30 Cases)
Defined in [`tests/golden_eval_dataset.py`](file:///F:/git/web_scraper_ai/tests/golden_eval_dataset.py):

| Category | Cases | Description |
| :--- | :---: | :--- |
| **Intent & Constraint Extraction** | 8 | Evaluates extraction of optimization goals, budgets, form factors (`UDIMM`/`SODIMM`), and capacities from natural language. |
| **Tool Selection & Argument Hygiene** | 6 | Evaluates correct tool selection, argument formatting (`query`, `max_price`, `capacity_gb`, `form_factor`), and avoidance of unpermitted tools. |
| **Product Ranking & Value Scoring** | 6 | Validates deterministic Value Scores, price-per-GB, budget ceilings, and categorical winners (`best_value`, `best_performance`, `budget_pick`). |
| **Grounding & Missing Information** | 6 | Checks that final recommendations quote exact prices and specs from scraped products without hallucinating. |
| **Edge Cases & Graceful Degradation** | 4 | Tests impossible budgets (e.g. ₹100), non-hardware queries, empty strings, and conflicting constraints. |

### Deterministic Evaluation Graders
Implemented in [`tests/eval_graders.py`](file:///F:/git/web_scraper_ai/tests/eval_graders.py):
- `grade_intent_extraction`: Verifies optimization, budget string/value, form factor, and capacity.
- `grade_tool_selection`: Validates permitted vs. forbidden tools and arguments.
- `grade_budget_compliance`: Asserts 100% of priced products respect the user's budget ceiling.
- `grade_form_factor_compliance`: Asserts 100% of products match the requested form factor.
- `grade_product_ranking`: Verifies deterministic ranking order and winner assignments.
- `grade_grounding`: Analyzes answer claims against tool outputs for spec and price veracity.
- `generate_report` & `format_markdown_report`: Compiles automated Markdown and JSON evaluation reports.

### Evaluation Status & Empirical Benchmark Results

We track two evaluation modes:
1. **Offline Deterministic Ground Truth (`pytest tests/test_golden_evaluation.py`)**: Asserts that deterministic constraint parsing, budget compliance, product ranking formulas, and grounding graders function accurately with 100% offline predictability.
2. **Live LLM Real-World Benchmark (`scripts/evaluate_live_llm.py`)**: Assesses real-world multi-step agent planning, tool execution, grounding, and edge-case handling against live model inference.

---

#### 1. Offline Deterministic Evaluation (Ground Truth Baseline)
*Report Artifact: `tests/output/golden_eval_report.md`*

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Total Test Cases** | **30 / 30** | All 30 cases across 5 categories |
| **Overall Accuracy** | **100.0%** (30 passed, 0 failed) | Clean deterministic pass |
| **Failure Rate** | **0.0%** | Zero logic or grading defects |
| **Average Latency** | **3.7ms** | Sub-millisecond unit evaluations |
| **Grounding Quality Score** | **100.0%** | Verified against structured HTML fixtures |

| Category | Total Cases | Passed | Failed | Accuracy | Avg Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **intent_extraction** | 8 | 8 | 0 | **100.0%** | **100.0%** |
| **tool_selection** | 6 | 6 | 0 | **100.0%** | **100.0%** |
| **product_ranking** | 6 | 6 | 0 | **100.0%** | **100.0%** |
| **grounding** | 6 | 6 | 0 | **100.0%** | **100.0%** |
| **edge_case** | 4 | 4 | 0 | **100.0%** | **100.0%** |

---

#### 2. Live LLM Real-World Benchmark Results (Full 30 Cases)
*Report Artifact: `tests/output/live_llm_eval_report.md`*

Empirical results from the full 30-case execution against the live LLM agent:

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Total Test Cases** | **30** | Full execution across all 5 benchmark categories |
| **Overall Accuracy** | **60.0%** (18 / 30 passed) | End-to-end task success against strict ground truth |
| **Failure Rate** | **40.0%** (12 / 30 failed) | Granular breakdown below |
| **Average Latency** | **153,575.7ms** (~153.6s) | Real-world multi-step iterative research loops |
| **Grounding Quality Score** | **63.7%** | Fact extraction and price veracity score |

##### Live Category Breakdown:
| Category | Total Cases | Passed | Failed | Accuracy | Avg Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **intent_extraction** | 8 | 8 | 0 | **100.0%** | **100.0%** | ✅ Clean Pass |
| **product_ranking** | 6 | 6 | 0 | **100.0%** | **100.0%** | ✅ Clean Pass |
| **tool_selection** | 6 | 2 | 4 | **33.3%** | **74.7%** | ⚠️ Partial Pass |
| **edge_case** | 4 | 1 | 3 | **25.0%** | **68.3%** | ⚠️ Partial Pass |
| **grounding** | 6 | 1 | 5 | **16.7%** | **53.1%** | ⚠️ Partial Pass |

##### Detailed Case Verdicts (All 30 Cases):
| ID | Category | Task | Status | Score | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `intent_01` | intent_extraction | Find me the best DDR5 RAM deal on Amazon under 10000 | ✅ PASS | `100%` | 1.4ms |
| `intent_02` | intent_extraction | I need 32GB laptop DDR5 memory | ✅ PASS | `100%` | 1.0ms |
| `intent_03` | intent_extraction | What's the cheapest DDR5 RAM available for my desktop? | ✅ PASS | `100%` | 1.1ms |
| `intent_04` | intent_extraction | I don't care about the price, give me the fastest DDR5 kit for overclocking | ✅ PASS | `100%` | 0.8ms |
| `intent_05` | intent_extraction | Top rated 16GB desktop RAM below 5000 | ✅ PASS | `100%` | 0.8ms |
| `intent_06` | intent_extraction | Find SODIMM DDR5 for my notebook below 8000 | ✅ PASS | `100%` | 0.7ms |
| `intent_07` | intent_extraction | Recommend the most reliable DDR5 RAM kit for a workstation PC | ✅ PASS | `100%` | 0.9ms |
| `intent_08` | intent_extraction | Find fastest DDR5 RAM under 50k for my gaming rig | ✅ PASS | `100%` | 0.8ms |
| `tool_01` | tool_selection | Search Amazon India for DDR5 6000MHz memory kits under 12000 | ❌ FAIL | `40%` | 284,525.4ms |
| `tool_02` | tool_selection | Inspect the operating voltage and timings of Corsair Vengeance DDR5 from Amazon | ✅ PASS | `100%` | 283,690.2ms |
| `tool_03` | tool_selection | Compare Corsair Vengeance and G.Skill Trident Z5 side by side | ❌ FAIL | `67%` | 301,290.1ms |
| `tool_04` | tool_selection | Scrape the hardware specification article at Wikipedia | ✅ PASS | `100%` | 32,816.7ms |
| `tool_05` | tool_selection | Find 64GB DDR5 memory modules for server virtualization | ❌ FAIL | `67%` | 48,715.0ms |
| `tool_06` | tool_selection | Search for SODIMM laptop RAM kits under 9000 | ❌ FAIL | `75%` | 361,561.9ms |
| `rank_01` | product_ranking | Find DDR5 RAM under 10000 and confirm all results respect the budget | ✅ PASS | `100%` | 287,489.9ms |
| `rank_02` | product_ranking | Find SODIMM DDR5 memory kits for a gaming laptop | ✅ PASS | `100%` | 299,034.1ms |
| `rank_03` | product_ranking | Identify the best value DDR5 RAM balancing speed and price | ✅ PASS | `100%` | 267,045.4ms |
| `rank_04` | product_ranking | Find the cheapest DDR5 memory module available | ✅ PASS | `100%` | 309,114.5ms |
| `rank_05` | product_ranking | Find the highest performance DDR5 gaming kit regardless of price | ✅ PASS | `100%` | 310,508.8ms |
| `rank_06` | product_ranking | Perform a multi-item comparison and verify verdicts | ✅ PASS | `100%` | 319,468.6ms |
| `ground_01` | grounding | What DDR5 RAM kits did you find on Amazon India? | ❌ FAIL | `60%` | 305,485.5ms |
| `ground_02` | grounding | How much does the Crucial 16GB DDR5 RAM cost on Amazon? | ✅ PASS | `100%` | 82,069.1ms |
| `ground_03` | grounding | Is the Crucial 32GB 5200MHz memory suitable for my desktop motherboard? | ❌ FAIL | `33%` | 38,930.1ms |
| `ground_04` | grounding | What is the price and specs of the Kingston FURY Beast 32GB RAM? | ❌ FAIL | `75%` | 293,076.0ms |
| `ground_05` | grounding | Explain why the top recommended RAM was chosen using its Value Score | ❌ FAIL | `0%` | 37,811.1ms |
| `ground_06` | grounding | What is the operating voltage and warranty on the Corsair Vengeance DDR5 kit? | ❌ FAIL | `50%` | 289,894.2ms |
| `edge_01` | edge_case | Find me 64GB DDR5 RAM for under 100 rupees | ✅ PASS | `100%` | 54,061.9ms |
| `edge_02` | edge_case | What is the weather in Mumbai today? | ❌ FAIL | `78%` | 18,031.5ms |
| `edge_03` | edge_case | [Empty prompt query] | ❌ FAIL | `67%` | 18,490.0ms |
| `edge_04` | edge_case | Find SODIMM DDR5 RAM for my standard desktop ATX motherboard | ❌ FAIL | `29%` | 364,153.0ms |

##### Failure Diagnostics & Empirical Observations:
- **`tool_01`, `tool_05`, `tool_06` (Tool Selection & Parameter Hygiene)**:
  - Model passed valid numeric parameters as string format (e.g. `max_price='12000'` instead of float `12000.0`, `capacity_gb='64'` instead of integer `64`).
  - `tool_01` autonomously called `scrape_url` to inspect extra product details when Amazon search returned a limited catalog.
- **`tool_03` (Comparison Chaining)**:
  - In direct side-by-side comparison, the agent initiated web searches to gather product URLs before comparison rather than reading them from single-step cache context.
- **`ground_01`, `ground_03`, `ground_04`, `ground_06` (Grounding & Exact Wording)**:
  - The model answered the technical questions correctly, but phrased answers conversationally rather than including exact literal keywords (`"SODIMM"`, `"laptop"`, `"unavailable"`), resulting in strict keyword-presence check failures.
- **`ground_05` (Value Score Grader)**:
  - Grader threw a string conversion error (`could not convert string to float: ''`) when extracting the score value from conversational prose.
- **`edge_02`, `edge_03`, `edge_04` (Edge Case Handling)**:
  - For out-of-domain tasks (e.g., weather queries), the agent politely answered that it cannot fetch weather, but missed specific canned phrases (`"shopping"`, `"unable"`).
  - For contradictory requests (`SODIMM for ATX desktop`), the agent took extensive time attempting to resolve motherboard memory slot compatibility.

---

### Running the Evaluation Suite

#### 1. Automated Offline Pytest Evaluation (Fast, No API calls):
```bash
uv run pytest tests/test_golden_evaluation.py -v
```

#### 2. Run a Single Test Case Offline:
```bash
uv run pytest tests/test_golden_evaluation.py -k "tool_01" -v -s
```

#### 3. Run Live LLM Evaluation Runner (`scripts/evaluate_live_llm.py`):
Benchmarking against the live LLM (Ollama or Groq):

```bash
# Run all 30 cases with live model
uv run python scripts/evaluate_live_llm.py --verbose

# Run a single case by ID
uv run python scripts/evaluate_live_llm.py --case tool_01 --verbose

# Run a specific category
uv run python scripts/evaluate_live_llm.py --category product_ranking --verbose

# Specify custom delay and steps
uv run python scripts/evaluate_live_llm.py --delay 2.0 --max-steps 3 --model qwen3.5:4b
```

---

## 🚦 Running the Server

Start the FastAPI application with Uvicorn:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to:
- **Interactive UI**: `http://localhost:8000/`
- **Swagger API Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📡 API Reference

### 1. Autonomous Research Task (`POST /api/research`)
Multi-step ReAct agent workflow: searches Amazon, inspects details, computes deterministic Value Scores, and synthesizes purchase recommendations.

- **Request Body**:
  ```json
  {
    "task": "Find me the best DDR5 RAM deals on Amazon under 50000",
    "max_steps": 6
  }
  ```
- **Response (200 OK)**:
  Returns structured candidate products, comparison metrics, deterministic rankings, and synthesized Markdown verdict.

---

### 2. General Agent Task (`POST /api/agent/task`)
Accepts a natural language task across all registered tools (`search_amazon`, `search_web`, `scrape_url`, `search_rag`, `extract_information`).

- **Request Body**:
  ```json
  {
    "task": "Find me DDR5 RAM deals on Amazon under 50000"
  }
  ```

---

### 3. Real-Time Tool Execution Stream (`POST /api/agent/stream`)
Server-Sent Events (SSE) streaming endpoint powering the web interface. Emits real-time events for every step: `planning`, `tool_start`, `tool_end`, `deterministic_ranking`, `synthesis_start`, and `final_response`.

---

### 4. Scrape & Ingestion Pipeline (`POST /api/scrape`)
Fetches a webpage, cleans HTML, splits into chunks, embeds with `google/embeddinggemma-2`, and stores in ChromaDB.

- **Request Body**:
  ```json
  {
    "url": "https://example.com",
    "force_refresh": false
  }
  ```

---

### 5. Question Answering RAG Pipeline (`POST /api/query`)
Executes the isolated QA pipeline against already indexed ChromaDB vectors.

- **Request Body**:
  ```json
  {
    "url": "https://example.com",
    "query": "What is the purpose of this domain?",
    "auto_scrape": true
  }
  ```

---

### 6. Health Status (`GET /health`)
Checks connectivity to Ollama / Groq, SentenceTransformers (`google/embeddinggemma-2`), and ChromaDB.

- **Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "ollama_service": "connected (qwen3.5:4b)",
    "embedding_service": "loaded (google/embeddinggemma-2)",
    "embedding_model": "google/embeddinggemma-2",
    "llm_model": "qwen3.5:4b"
  }
  ```

---

## 🛡️ Anti-Bot Protection & Edge Cases

- **Amazon AWS WAF & Cloudflare**: Standard HTTP clients (Requests, httpx, urllib) fail with `HTTP 503 Service Unavailable` due to mismatched OpenSSL TLS/JA3 fingerprints. We use `curl-cffi` with authentic Chrome TLS impersonation and automatic exponential retry backoff.
- **Login-Walled Sites (LinkedIn)**: URLs behind mandatory user authentication return `HTTP 999 Request Denied`. The scraper catches this and raises a clean `403 Forbidden` response explaining the authentication wall.

---

## 🧪 Running Automated Tests

Run the complete test suite:

```bash
# Run all 56 offline tests
uv run pytest -m "not smoke"

# Run with verbose output
uv run pytest -v
```

### Test Suite Structure:
- `tests/test_golden_evaluation.py`: 31 tests validating the 30-case golden dataset against deterministic graders.
- `tests/test_deterministic.py`: 7 tests validating Value Score formula, sorting algorithms, and side-by-side matrices.
- `tests/test_reliability_fault_injection.py`: 7 tests verifying repetition interceptors, malformed inputs, and quota safeguards.
- `tests/test_stream.py`: 4 tests validating SSE streaming events and Web UI routes.
- `tests/test_app.py`: 2 tests verifying ChromaDB ingestion and RAG QA pipelines.
- `tests/test_agent.py`: 2 tests verifying AgentService tool dispatching.
- `tests/test_multi_step_agent.py`: 3 tests verifying multi-step iterative research loops.

---

## 📁 Project Directory Structure

```text
web_scraper_ai/
├── .env                              # Local credentials & Ollama/Groq configuration
├── .env.example                      # Template environment variables
├── .gitignore                        # Git exclusion rules
├── pyproject.toml                    # Dependencies, pytest config & project metadata
├── README.md                         # Project documentation
├── main.py                           # Application entrypoint (uvicorn main:app)
│
├── app/                              # Core Application Package
│   ├── __init__.py                   # Package exports
│   ├── main.py                       # FastAPI application & SSE streaming endpoints
│   ├── models.py                     # Pydantic schemas (Product, ProductDetail, etc.)
│   ├── agent/                        # Autonomous Hardware Agent
│   │   ├── __init__.py
│   │   └── service.py                # AgentService, value scoring, tools, & Ollama loop
│   ├── rag/                          # RAG & ChromaDB Pipeline
│   │   ├── __init__.py
│   │   ├── vector_store.py           # ChromaDB manager & GemmaEmbeddings
│   │   ├── ingestion.py              # Scrape -> Chunk -> Embed -> ChromaDB
│   │   └── qa.py                     # Vector retrieval & answer synthesis
│   └── scrapers/                     # Web Scraping & TLS Impersonation
│       ├── __init__.py
│       ├── base_scraper.py           # TLS anti-bot web scraper (curl-cffi)
│       └── amazon_scraper.py         # Structured Amazon search & details
│
├── scripts/                          # Evaluation & Tooling Scripts
│   └── evaluate_live_llm.py          # Live LLM evaluation benchmark runner
│
├── static/                           # Interactive Web UI Frontend
│   ├── index.html                    # Dark glassmorphism dashboard
│   ├── styles.css                    # Styling & pulsing animations
│   └── app.js                        # SSE streaming listener & rendering
│
└── tests/                            # Comprehensive Automated Test Suites
    ├── __init__.py
    ├── conftest.py                   # Mock data fixtures & mock LLM implementations
    ├── golden_eval_dataset.py        # 30 curated golden evaluation test cases
    ├── eval_graders.py               # Deterministic graders & markdown reporting
    ├── test_golden_evaluation.py     # 31 tests running golden dataset evaluation
    ├── test_deterministic.py         # Deterministic ranking & Value Score tests
    ├── test_reliability_fault_injection.py # Repetition & error handling tests
    ├── test_app.py                   # Ingestion, RAG, and API tests
    ├── test_stream.py                # Streaming SSE & UI telemetry tests
    ├── test_agent.py                 # Agent tool invocation tests
    ├── test_multi_step_agent.py      # Multi-step research loop tests
    └── output/                       # Evaluation reports
        ├── golden_eval_report.md     # Latest golden evaluation markdown report
        └── golden_eval_report.json   # Machine-readable evaluation report
```

---

## 📄 License

Distributed under the [MIT License](LICENSE). Feel free to use, modify, and distribute for personal and commercial projects.
