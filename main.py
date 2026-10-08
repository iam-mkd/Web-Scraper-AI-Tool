import os
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables on startup (LangSmith tracing, etc.)
load_dotenv()

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, HttpUrl, Field
import httpx

from vector_store import get_vector_store_manager
from ingestion import IngestionService
from qa import QAService
from agent_service import get_agent_service
from models import Product, ResearchRequest, ResearchResponse

app = FastAPI(
    title="Web Scraper AI API",
    description=(
        "Decoupled Web Scraper and RAG API: "
        "Ingestion Pipeline (fetch -> clean -> chunk -> embed -> ChromaDB) & "
        "QA Pipeline (embed question -> ChromaDB -> top-k chunks -> Groq LLM -> answer)"
        "Autonomous AI Web Scraper and RAG API: "
        "Agent Task Execution (tools: Amazon India search, scraping, RAG) & "
        "Decoupled Ingestion & QA Pipelines."
    ),
    version="3.0.0",
)

# Initialize distinct services
vector_store_manager = get_vector_store_manager()
ingestion_service = IngestionService(vector_store_manager=vector_store_manager)
qa_service = QAService(vector_store_manager=vector_store_manager)
agent_service = get_agent_service()


# ---------------------------------------------------------
# Exception Handlers to guarantee strict JSON-only responses
# ---------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if isinstance(exc.detail, str) else "HTTP Error",
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
        },
    )


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------
class ScrapeRequest(BaseModel):
    url: HttpUrl = Field(..., description="The webpage URL to fetch, clean, chunk, embed, and store")
    force_refresh: bool = Field(
        default=False,
        description="If True, re-scrapes and updates existing ChromaDB embeddings for this URL",
    )


class ScrapeResponse(BaseModel):
    status: str
    url: str
    title: str
    chunks_indexed: int
    char_count: Optional[int] = None
    sample_chunk: Optional[str] = None
    message: str


class QueryRequest(BaseModel):
    url: HttpUrl = Field(..., description="The webpage URL to query against")
    query: str = Field(..., min_length=1, description="Question to ask about the webpage content")
    top_k: Optional[int] = Field(
        default=None,
        description="Number of chunks to retrieve. If None, dynamically retrieves all chunks for pages with <= 12 chunks, or 8 chunks for larger pages.",
    )
    auto_scrape: bool = Field(
        default=True,
        description="If True, automatically triggers the ingestion workflow if the URL is not yet indexed",
    )
    force_refresh: bool = Field(
        default=False,
        description="If True, re-ingests the URL even if already indexed before answering",
    )


class SourceMetadata(BaseModel):
    url: str
    chunk_id: int
    title: Optional[str] = None
    total_chunks: Optional[int] = None


class SourceItem(BaseModel):
    content: str
    metadata: SourceMetadata
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    url: Optional[str] = None
    query: Optional[str] = None
    model: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    groq_service: str
    embedding_service: Optional[str] = "loaded (google/embeddinggemma-2)"
    ollama_service: Optional[str] = "migrated to sentence-transformers (google/embeddinggemma-2)"
    embedding_model: str
    llm_model: str
    langsmith_tracing: bool
    langsmith_project: Optional[str]


class AgentTaskRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=2,
        description="Natural language task or command for the agent (e.g. 'Find me DDR5 RAM deals on Amazon under 10000')",
    )


class AgentTaskResponse(BaseModel):
    task: str
    tool_called: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None
    products: List[Dict[str, Any]] = []
    explanation: str
    model: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


@app.get("/", include_in_schema=False)
async def serve_root(request: Request):
    accept = request.headers.get("accept", "")
    # If client explicitly asks for application/json and not HTML, return API metadata
    if "application/json" in accept and "text/html" not in accept:
        return {
            "name": "Web Scraper AI Tool API",
            "description": "Autonomous Agent Research, Tool Execution & Decoupled Ingestion/QA Pipelines",
            "workflows": {
                "research": "natural language task -> GPT-OSS 120B -> search_amazon -> retrieve products -> extract Product models -> filter/compare -> final answer (POST /api/research)",
                "agent": "natural language task -> LLM selects tool -> tool executes -> deterministic Python logic -> LLM explanation (POST /api/agent/task)",
                "stream": "real-time SSE tool execution traces (POST /api/agent/stream)",
                "ingestion": "fetch webpage -> clean html -> chunk -> embed -> ChromaDB (POST /api/scrape)",
                "qa": "embed question -> ChromaDB -> top-k chunks -> Groq LLM -> answer (POST /api/query)",
            },
        }

    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(content={"message": "Web Scraper AI API is running. UI not found in static/."})


@app.get("/api", response_class=JSONResponse)
async def api_info():
    return {
        "name": "Web Scraper AI Tool API",
        "description": "Autonomous Agent Research, Tool Execution & Decoupled Ingestion/QA Pipelines",
        "workflows": {
            "research": "natural language task -> GPT-OSS 120B -> search_amazon -> retrieve products -> extract Product models -> filter/compare -> final answer (POST /api/research)",
            "agent": "natural language task -> LLM selects tool -> tool executes -> deterministic Python logic -> LLM explanation (POST /api/agent/task)",
            "stream": "real-time SSE tool execution traces (POST /api/agent/stream)",
            "ingestion": "fetch webpage -> clean html -> chunk -> embed -> ChromaDB (POST /api/scrape)",
            "qa": "embed question -> ChromaDB -> top-k chunks -> Groq LLM -> answer (POST /api/query)",
        },
        "endpoints": {
            "GET /": "Interactive HTML/CSS/JS UI Frontend",
            "POST /api/agent/stream": "Real-time Server-Sent Events (SSE) tool execution traces",
            "POST /api/research": "Execute autonomous shopping research with structured Product schema & deterministic comparison",
            "POST /api/agent/task": "Execute autonomous agent task with tool selection & structured data",
            "POST /api/scrape": "Execute ingestion pipeline for a webpage",
            "POST /api/query": "Execute question answering pipeline against indexed webpage",
            "GET /health": "Check system and model health status",
            "GET /docs": "Interactive Swagger UI documentation",
        },
    }



@app.get("/health", response_model=HealthResponse, response_class=JSONResponse)
async def health_check():
    # Check Embedding service (google/embeddinggemma-2 via sentence-transformers)
    embed_ok = True
    embed_status_str = "loaded (google/embeddinggemma-2)"
    try:
        if not hasattr(vector_store_manager, "embeddings") or vector_store_manager.embeddings.model is None:
            embed_ok = False
            embed_status_str = "model not initialized"
    except Exception as e:
        embed_ok = False
        embed_status_str = f"error ({str(e)})"

    # Check Groq API service
    groq_key = os.getenv("GROQ_API_KEY")
    groq_ok = False
    groq_status_str = "missing GROQ_API_KEY in .env"
    if groq_key:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {groq_key}"},
                )
                if resp.status_code == 200:
                    groq_ok = True
                    groq_status_str = "connected"
                else:
                    groq_status_str = f"API error (HTTP {resp.status_code})"
        except Exception as e:
            groq_status_str = f"unreachable ({str(e)})"

    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project_name = os.getenv("LANGCHAIN_PROJECT")
    current_llm_model = getattr(qa_service, "llm_model", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

    is_healthy = embed_ok and groq_ok

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        groq_service=groq_status_str,
        embedding_service=embed_status_str,
        ollama_service="migrated to sentence-transformers (google/embeddinggemma-2)",
        embedding_model="google/embeddinggemma-2",
        llm_model=current_llm_model,
        langsmith_tracing=tracing_enabled,
        langsmith_project=project_name,
    )


@app.post("/api/scrape", response_model=ScrapeResponse, response_class=JSONResponse)
async def scrape_endpoint(request: ScrapeRequest):
    """
    Ingestion Pipeline:
    fetch webpage -> clean html -> chunk -> embed -> chromadb
    """
    url_str = str(request.url)
    try:
        result = await ingestion_service.ingest_url(
            url=url_str,
            force_refresh=request.force_refresh,
        )
        return ScrapeResponse(
            status=result["status"],
            url=result["url"],
            title=result["title"],
            chunks_indexed=result["chunks_indexed"],
            char_count=result.get("char_count"),
            sample_chunk=result.get("sample_chunk"),
            message=result["message"],
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch webpage: HTTP {e.response.status_code}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ingestion failed: {str(e)}",
        )


@app.post("/api/query", response_model=QueryResponse, response_class=JSONResponse)
async def query_endpoint(request: QueryRequest):
    """
    Question Answering Pipeline:
    embed question -> chroma db -> top k chunks -> granite (LLM) -> answer

    Convenience behavior:
    If URL is not yet indexed, optionally runs ingestion first if auto_scrape=True.
    """
    url_str = str(request.url)

    # 1. Ingestion check / Auto-ingestion
    is_indexed = vector_store_manager.is_url_indexed(url_str)
    if not is_indexed or request.force_refresh:
        if not request.auto_scrape:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webpage {url_str} is not yet indexed in ChromaDB, and auto_scrape is disabled.",
            )

        # Run ingestion workflow first
        try:
            await ingestion_service.ingest_url(
                url=url_str,
                force_refresh=request.force_refresh,
            )
        except PermissionError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Auto-ingestion failed before querying: {str(e)}",
            )

    # 2. Question Answering workflow
    try:
        qa_result = await qa_service.answer_query(
            url=url_str,
            query=request.query,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QA pipeline failed: {str(e)}",
        )

    return QueryResponse(
        answer=qa_result["answer"],
        sources=qa_result["sources"],
        url=qa_result["url"],
        query=qa_result["query"],
        model=qa_result["model"],
    )


@app.post("/api/agent/task", response_model=AgentTaskResponse, response_class=JSONResponse)
async def agent_task_endpoint(request: AgentTaskRequest):
    """
    Autonomous Agent Task Execution:
    task -> LLM selects tool -> tool execution (structured data) -> deterministic Python logic -> LLM explanation
    """
    try:
        result = await agent_service.run_task(task=request.task)
        return AgentTaskResponse(
            task=result["task"],
            tool_called=result.get("tool_called"),
            tool_arguments=result.get("tool_arguments"),
            tools_called=result.get("tools_called"),
            steps_taken=result.get("steps_taken", []),
            stats=result.get("stats"),
            products=result.get("products", []),
            detailed_products=result.get("detailed_products", []),
            comparison=result.get("comparison"),
            explanation=result["explanation"],
            model=result["model"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent task execution failed: {str(e)}",
        )


@app.post("/api/research", response_model=ResearchResponse, response_class=JSONResponse)
async def research_endpoint(request: ResearchRequest):
    """
    Autonomous Research Endpoint:
    Task -> GPT-OSS 120B multi-step loop -> search_amazon -> get_product_details -> compare_products -> final synthesis
    """
    try:
        result = await agent_service.run_task(task=request.task)
        parsed_products: List[Product] = []
        for p in result.get("products", []):
            try:
                parsed_products.append(Product.model_validate(p))
            except Exception:
                pass

        return ResearchResponse(
            task=result["task"],
            tool_called=result.get("tool_called"),
            tool_arguments=result.get("tool_arguments"),
            tools_called=result.get("tools_called", []),
            steps_taken=result.get("steps_taken", []),
            stats=result.get("stats"),
            products=parsed_products,
            detailed_products=result.get("detailed_products", []),
            comparison=result.get("comparison"),
            final_answer=result.get("final_answer") or result.get("explanation", ""),
            model=result["model"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research task execution failed: {str(e)}",
        )


class StreamTaskRequest(BaseModel):
    task: str = Field(..., min_length=2, description="Natural language task or command for the agent")


@app.post("/api/agent/stream")
async def agent_stream_post_endpoint(request: StreamTaskRequest):
    """
    Server-Sent Events (SSE) streaming endpoint for real-time tool execution traces.
    Streams granular events as the agent reasons, invokes tools, computes rankings, and synthesizes answers.
    """
    async def event_generator():
        try:
            async for event in agent_service.run_task_stream(task=request.task):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/agent/stream")
async def agent_stream_get_endpoint(task: str):
    """
    GET Server-Sent Events endpoint for easy browser EventSource testing.
    """
    async def event_generator():
        try:
            async for event in agent_service.run_task_stream(task=task):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

