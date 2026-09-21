import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from scraper import scrape_webpage
from rag_service import RAGService
from fastapi.testclient import TestClient
from main import app


async def test_scraper_and_rag():
    print("=== 1. Testing Scraper ===")
    test_url = "https://example.com"
    scraped = await scrape_webpage(test_url)
    print(f"Title: {scraped['title']}")
    print(f"Content length: {len(scraped['content'])} chars")
    print(f"Content preview: {scraped['content'][:150]}...")
    assert len(scraped["content"]) > 0

    print("\n=== 2. Testing RAG Service ===")
    rag = RAGService(persist_directory="./test_chroma_db", collection_name="test_docs")
    indexed_count = rag.index_webpage(
        url=test_url,
        title=scraped["title"],
        content=scraped["content"],
        force_refresh=True,
    )
    print(f"Indexed chunks: {indexed_count}")
    assert indexed_count > 0

    print("\n=== 3. Testing RAG Query with Ollama ===")
    query = "What is the purpose of this domain?"
    result = await rag.query_webpage(url=test_url, query=query)
    print(f"Query: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Sources retrieved: {len(result['sources'])}")
    assert len(result["answer"]) > 0


def test_api():
    print("\n=== 4. Testing FastAPI Endpoints ===")
    client = TestClient(app)

    # Test /health
    health_resp = client.get("/health")
    print(f"Health Response ({health_resp.status_code}): {health_resp.json()}")
    assert health_resp.status_code == 200

    # Test /api/scrape
    scrape_payload = {"url": "https://example.com", "force_refresh": True}
    scrape_resp = client.post("/api/scrape", json=scrape_payload)
    print(f"Scrape Response ({scrape_resp.status_code}): {scrape_resp.json()}")
    assert scrape_resp.status_code == 200

    # Test /api/query
    query_payload = {
        "url": "https://example.com",
        "query": "What is this domain used for?",
        "auto_scrape": True
    }
    query_resp = client.post("/api/query", json=query_payload)
    print(f"Query Response ({query_resp.status_code}): {query_resp.json()}")
    assert query_resp.status_code == 200
    res_data = query_resp.json()
    assert "answer" in res_data
    assert "sources" in res_data
    print("\nAll automated tests PASSED successfully!")


if __name__ == "__main__":
    asyncio.run(test_scraper_and_rag())
    test_api()

