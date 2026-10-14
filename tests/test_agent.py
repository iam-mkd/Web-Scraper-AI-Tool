import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import app
from app.agent.service import (
    search_amazon,
    search_web,
    calculate_value_score,
    AgentService,
)
from tests.conftest import MockChatGroq, MOCK_AMAZON_SEARCH_HTML, FIXED_PRODUCTS_DATASET


@pytest.mark.asyncio
async def test_agent_direct():
    print("\n=== 1. Testing Search Tool Logic Directly ===")
    mock_products = list(FIXED_PRODUCTS_DATASET)

    with patch("app.scrapers.amazon_scraper.fetch_html", return_value=MOCK_AMAZON_SEARCH_HTML):
        result_str = await search_amazon.ainvoke({"query": "DDR5 RAM", "max_price": 50000.0})
        result = json.loads(result_str)

    assert result["status"] == "success"
    assert "cheapest_product" in result["stats"]
    assert "best_value_product" in result["stats"]
    assert len(result["products"]) > 0

    best_val = result["stats"]["best_value_product"]
    print(f"Top Recommended Value RAM: {best_val['name']}")
    print(f"Value Score: {best_val['value_score']}/100 | Price: ₹{best_val['price']}")


def test_agent_api():
    print("\n=== 2. Testing FastAPI Agent Endpoint (Mock) ===")
    client = TestClient(app)
    payload = {"task": "Find me the best DDR5 RAM deal on Amazon under 50000"}

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_amazon",
                "args": {"query": "DDR5 RAM", "max_price": 50000.0},
                "id": "api_call_1",
            }
        ],
    )
    final_synth = AIMessage(
        content="API recommendation: Corsair Vengeance 32GB DDR5.",
        tool_calls=[],
    )
    mock_llm = MockChatGroq([tool_call, final_synth])
    mock_agent = AgentService(llm=mock_llm)

    with patch("app.main.agent_service", mock_agent), \
         patch("app.scrapers.amazon_scraper.fetch_html", return_value=MOCK_AMAZON_SEARCH_HTML):
        response = client.post("/api/agent/task", json=payload)

    assert response.status_code == 200, f"API error: {response.text}"
    data = response.json()
    assert data.get("tool_called") in ("search_web", "search_amazon")
    assert len(data.get("products", [])) > 0
    assert data.get("explanation")
    print("\n>>> FastAPI Agent Endpoint Test Passed! <<<\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent_direct())
    test_agent_api()
