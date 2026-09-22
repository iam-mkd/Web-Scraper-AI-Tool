import asyncio
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

from scraper import scrape_webpage
from vector_store import VectorStoreManager
from ingestion import IngestionService
from qa import QAService
from fastapi.testclient import TestClient
from main import app

TEST_DIR = "./test_chroma_db"
TEST_COLL = "test_docs"


async def test_pipelines():
    test_url = "https://example.com"

    print("=== 1. Testing Scraper ===")
    scraped = await scrape_webpage(test_url)
    print(f"Title: {scraped['title']}")
    print(f"Content length: {len(scraped['content'])} chars")
    assert len(scraped["content"]) > 0

    print("\n=== 2. Testing Ingestion Pipeline (fetch -> clean -> chunk -> embed -> ChromaDB) ===")
    test_store = VectorStoreManager(persist_directory=TEST_DIR, collection_name=TEST_COLL)
    ingestion_service = IngestionService(vector_store_manager=test_store)

    ingest_result = await ingestion_service.ingest_url(test_url, force_refresh=True)
    print(f"Ingest Status: {ingest_result['status']}")
    print(f"Chunks Indexed: {ingest_result['chunks_indexed']}")
    print(f"Sample Chunk: {ingest_result.get('sample_chunk', '')[:100]}...")
    assert ingest_result["chunks_indexed"] > 0
    assert test_store.is_url_indexed(test_url) is True

    print("\n=== 3. Testing QA Pipeline (embed question -> ChromaDB -> top-k -> Granite LLM -> answer) ===")
    qa_service = QAService(vector_store_manager=test_store)
    query = "What is this domain used for?"
    qa_result = await qa_service.answer_query(url=test_url, query=query)
    print(f"Query: {qa_result['query']}")
    print(f"Answer: {qa_result['answer']}")
    print(f"Sources Count: {len(qa_result['sources'])}")
    assert len(qa_result["answer"]) > 0
    assert len(qa_result["sources"]) > 0


def test_api():
    print("\n=== 4. Testing Decoupled FastAPI Endpoints ===")
    client = TestClient(app)

    # Test /health
    health_resp = client.get("/health")
    print(f"Health Response ({health_resp.status_code}): {health_resp.json()}")
    assert health_resp.status_code == 200

    # Test POST /api/scrape
    scrape_payload = {"url": "https://example.com", "force_refresh": True}
    scrape_resp = client.post("/api/scrape", json=scrape_payload)
    print(f"Scrape Response ({scrape_resp.status_code}): {scrape_resp.json()}")
    assert scrape_resp.status_code == 200
    scrape_data = scrape_resp.json()
    assert scrape_data["status"] in ("success", "already_indexed")
    assert scrape_data["chunks_indexed"] > 0

    # Test POST /api/query
    query_payload = {
        "url": "https://example.com",
        "query": "What is the purpose of this domain?",
        "auto_scrape": True,
    }
    query_resp = client.post("/api/query", json=query_payload)
    print(f"Query Response ({query_resp.status_code}): {query_resp.json()}")
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
