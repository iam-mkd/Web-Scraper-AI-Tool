import os
import sys
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables on startup
load_dotenv()

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, HttpUrl, Field
import httpx

from scraper import scrape_webpage
from rag_service import RAGService

app = FastAPI(
    title="Web Scraper AI API",
    description="API for scraping webpages, chunking, embedding into ChromaDB, and querying with Ollama and LangSmith tracing.",
    version="1.0.0",
)

# Initialize RAG Service
rag_service = RAGService()


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
    url: HttpUrl = Field(..., description="The webpage URL to scrape and embed")
    force_refresh: bool = Field(
        default=False,
        description="If True, re-scrapes and updates existing ChromaDB embeddings for this URL",
    )


class ScrapeResponse(BaseModel):
    status: str
    url: str
    title: str
    chunks_indexed: int
    message: str


class QueryRequest(BaseModel):
    url: HttpUrl = Field(..., description="The webpage URL to query against")
    query: str = Field(..., min_length=1, description="Question to ask about the webpage content")
    auto_scrape: bool = Field(
        default=True,
        description="If True, automatically scrapes and embeds the webpage if not already indexed",
    )
    force_refresh: bool = Field(
        default=False,
        description="If True, re-scrapes even if already indexed before answering query",
    )


class QueryResponse(BaseModel):
    url: str
    query: str
    answer: str
    sources: List[str]
    model: str


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
        "description": "Scrape, chunk, embed into ChromaDB, and query with Ollama + LangSmith",
        "endpoints": {
            "POST /api/query": "Query a webpage with automatic scraping and RAG generation",
            "POST /api/scrape": "Explicitly scrape and index a webpage into ChromaDB",
            "GET /health": "Check system and service health status",
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
async def scrape_and_index(request: ScrapeRequest):
    url_str = str(request.url)

    # Check if already indexed and force_refresh is False
    if rag_service.is_url_indexed(url_str) and not request.force_refresh:
        return ScrapeResponse(
            status="already_indexed",
            url=url_str,
            title="Already Indexed",
            chunks_indexed=rag_service.index_webpage(url_str, "", "", force_refresh=False),
            message="Webpage is already indexed in ChromaDB. Set force_refresh=True to re-index.",
        )

    # Scrape webpage
    try:
        scraped_data = await scrape_webpage(url_str)
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
            detail=f"Error scraping webpage: {str(e)}",
        )

    # Chunk and index into ChromaDB
    try:
        num_chunks = rag_service.index_webpage(
            url=url_str,
            title=scraped_data["title"],
            content=scraped_data["content"],
            force_refresh=request.force_refresh,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embeddings or storing in ChromaDB: {str(e)}",
        )

    return ScrapeResponse(
        status="success",
        url=url_str,
        title=scraped_data["title"],
        chunks_indexed=num_chunks,
        message=f"Successfully scraped, chunked, and stored {num_chunks} chunks in ChromaDB.",
    )


@app.post("/api/query", response_model=QueryResponse, response_class=JSONResponse)
async def query_webpage_endpoint(request: QueryRequest):
    url_str = str(request.url)

    # Handle auto-scraping if URL is not yet indexed or force_refresh is True
    if not rag_service.is_url_indexed(url_str) or request.force_refresh:
        if not request.auto_scrape:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webpage {url_str} is not yet indexed in ChromaDB, and auto_scrape is disabled.",
            )

        # Scrape and index
        try:
            scraped_data = await scrape_webpage(url_str)
            rag_service.index_webpage(
                url=url_str,
                title=scraped_data["title"],
                content=scraped_data["content"],
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
                detail=f"Failed to scrape and index webpage before querying: {str(e)}",
            )

    # Execute RAG query (with LangSmith tracing)
    try:
        result = await rag_service.query_webpage(url=url_str, query=request.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query with LLM: {str(e)}",
        )

    return QueryResponse(
        url=url_str,
        query=request.query,
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
