import asyncio
import os
import shutil
import sys
import pytest
from unittest.mock import patch, AsyncMock
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from app.rag.vector_store import VectorStoreManager
from app.rag.ingestion import IngestionService
from app.rag.qa import QAService
from fastapi.testclient import TestClient
from app.main import app

TEST_DIR = os.path.join(PROJECT_ROOT, "test_chroma_db")
TEST_COLL = "test_docs"

MOCK_SCRAPED = {
    "url": "https://example.com",
    "title": "Example Domain",
    "content": "Example Domain. This domain is established to be used for illustrative examples in documents.",
    "char_count": 95,
}


@pytest.mark.asyncio
async def test_pipelines():
    test_url = "https://example.com"

    print("=== 1. Testing Scraper (Mocked Webpage) ===")
    with patch("app.scrapers.base_scraper.fetch_html", return_value="<html><head><title>Example Domain</title></head><body><p>Example Domain. Illustrative examples.</p></body></html>"):
        from app.scrapers.base_scraper import scrape_webpage
        scraped = await scrape_webpage(test_url)
    assert len(scraped["content"]) > 0

    print("\n=== 2. Testing Ingestion Pipeline with Fast FakeEmbeddings ===")
    fake_embeddings = FakeEmbeddings(size=768)
    test_store = VectorStoreManager(
        persist_directory=TEST_DIR,
        collection_name=TEST_COLL,
        embeddings=fake_embeddings,
    )
    ingestion_service = IngestionService(vector_store_manager=test_store)

    with patch("app.rag.ingestion.scrape_webpage", return_value=MOCK_SCRAPED):
        ingest_result = await ingestion_service.ingest_url(test_url, force_refresh=True)

    assert ingest_result["chunks_indexed"] > 0
    assert test_store.is_url_indexed(test_url) is True

    print("\n=== 3. Testing QA Pipeline with Mock LLM ===")
    qa_service = QAService(vector_store_manager=test_store)
    query = "What is this domain used for?"

    fake_llm = FakeListChatModel(responses=["This domain is used for illustrative examples in documents."])
    with patch.object(qa_service, "get_llm", return_value=fake_llm):
        qa_result = await qa_service.answer_query(url=test_url, query=query)
        assert len(qa_result["answer"]) > 0
        assert len(qa_result["sources"]) > 0


def test_api():
    print("\n=== 4. Testing Decoupled FastAPI Endpoints ===")
    client = TestClient(app)

    # Test /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json().get("embedding_model") == "google/embeddinggemma-2"

    # Test POST /api/scrape with mocked ingestion
    fake_ingest_res = {
        "status": "success",
        "url": "https://example.com",
        "title": "Example Domain",
        "chunks_indexed": 1,
        "char_count": 95,
        "sample_chunk": "Example Domain content",
        "message": "Successfully indexed 1 chunks into ChromaDB.",
    }
    with patch("app.main.ingestion_service.ingest_url", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = fake_ingest_res
        scrape_payload = {"url": "https://example.com", "force_refresh": True}
        scrape_resp = client.post("/api/scrape", json=scrape_payload)
        assert scrape_resp.status_code == 200
        scrape_data = scrape_resp.json()
        assert scrape_data["status"] == "success"
        assert scrape_data["chunks_indexed"] > 0

    # Test POST /api/query with mocked QA
    fake_qa_res = {
        "answer": "This domain is used for illustrative documentation.",
        "sources": [
            {
                "content": "Example domain text",
                "metadata": {"url": "https://example.com", "chunk_id": 0},
                "score": 0.85,
            }
        ],
        "url": "https://example.com",
        "query": "What is this domain?",
        "model": "openai/gpt-oss-120b",
    }
    with patch("app.main.vector_store_manager.is_url_indexed", return_value=True), \
         patch("app.main.qa_service.answer_query", new_callable=AsyncMock) as mock_qa:
        mock_qa.return_value = fake_qa_res
        query_payload = {"url": "https://example.com", "query": "What is the purpose of this domain?"}
        query_resp = client.post("/api/query", json=query_payload)
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        assert "answer" in query_data
        assert len(query_data["sources"]) > 0

    print("\nAll automated decoupled tests PASSED successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(test_pipelines())
        test_api()
    finally:
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR, ignore_errors=True)
