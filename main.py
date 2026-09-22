import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables on startup (LangSmith tracing, etc.)
load_dotenv()

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, HttpUrl, Field
import httpx

from vector_store import get_vector_store_manager
from ingestion import IngestionService
from qa import QAService

app = FastAPI(
    title="Web Scraper AI API",
    description=(
        "Decoupled Web Scraper and RAG API: "
        "Ingestion Pipeline (fetch -> clean -> chunk -> embed -> ChromaDB) & "
        "QA Pipeline (embed question -> ChromaDB -> top-k chunks -> Granite LLM -> answer)"
    ),
    version="2.0.0",
)

# Initialize distinct services
vector_store_manager = get_vector_store_manager()
ingestion_service = IngestionService(vector_store_manager=vector_store_manager)
qa_service = QAService(vector_store_manager=vector_store_manager)


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
    ollama_service: str
    embedding_model: str
    llm_model: str
    langsmith_tracing: bool
    langsmith_project: Optional[str]


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "name": "Web Scraper AI Tool API",
        "description": "Decoupled Ingestion Pipeline and Question Answering Pipeline",
        "workflows": {
            "ingestion": "fetch webpage -> clean html -> chunk -> embed -> ChromaDB (POST /api/scrape)",
            "qa": "embed question -> ChromaDB -> top-k chunks -> Granite LLM -> answer (POST /api/query)",
        },
        "endpoints": {
            "POST /api/scrape": "Execute ingestion pipeline for a webpage",
            "POST /api/query": "Execute question answering pipeline against indexed webpage",
            "GET /health": "Check system and model health status",
            "GET /docs": "Interactive Swagger UI documentation",
        },
    }


@app.get("/health", response_model=HealthResponse, response_class=JSONResponse)
async def health_check():
    # Check Ollama service
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project_name = os.getenv("LANGCHAIN_PROJECT")

    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_service="connected" if ollama_ok else "unreachable (ensure Ollama is running)",
        embedding_model="nomic-embed-text",
        llm_model="granite3-dense:2b",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
