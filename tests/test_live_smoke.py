import os
import pytest
from dotenv import load_dotenv

load_dotenv()

RUN_LIVE = os.getenv("RUN_LIVE_TESTS", "0").lower() in ("1", "true", "yes")


@pytest.mark.smoke
@pytest.mark.skipif(not RUN_LIVE, reason="Live smoke tests disabled by default. Set RUN_LIVE_TESTS=1 to run.")
@pytest.mark.asyncio
async def test_live_amazon_integration():
    """Live smoke test: verify Amazon scraper against real amazon.in."""
    from app.scrapers.amazon_scraper import search_amazon_products
    products = await search_amazon_products("DDR5 RAM", max_attempts=2)
    assert isinstance(products, list)
    if products:
        assert "name" in products[0]
        assert "url" in products[0]


@pytest.mark.smoke
@pytest.mark.skipif(not RUN_LIVE or not os.getenv("GROQ_API_KEY"), reason="Live smoke tests require RUN_LIVE_TESTS=1 and GROQ_API_KEY.")
@pytest.mark.asyncio
async def test_live_groq_integration():
    """Live smoke test: verify real Groq API connectivity and tool invocation."""
    from app.agent.service import get_agent_service
    agent = get_agent_service()
    llm = agent.get_llm()
    resp = await llm.ainvoke("Say hello in one word.")
    assert resp.content
    assert len(str(resp.content).strip()) > 0


@pytest.mark.smoke
def test_live_api_health_endpoint():
    """Live smoke test: verify FastAPI /health endpoint."""
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["embedding_model"] == "google/embeddinggemma-2"
