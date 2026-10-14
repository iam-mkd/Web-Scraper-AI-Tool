import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import app
from app.agent.service import (
    search_amazon,
    get_product_details,
    compare_products,
    AgentService,
)
from tests.conftest import (
    MockChatGroq,
    MOCK_AMAZON_SEARCH_HTML,
    MOCK_AMAZON_DETAIL_HTML,
    FIXED_PRODUCTS_DATASET,
)


@pytest.mark.asyncio
async def test_tools_directly():
    print("\n=== 1. Testing Search Tool ===")
    with patch("app.scrapers.amazon_scraper.fetch_html", return_value=MOCK_AMAZON_SEARCH_HTML):
        search_res_str = await search_amazon.ainvoke({"query": "DDR5 RAM", "max_price": 50000.0})
        search_res = json.loads(search_res_str)

    assert search_res["status"] == "success"
    prods = search_res["products"]
    assert len(prods) > 0

    print("\n=== 2. Testing Deep Product Detail Extraction Tool ===")
    top_prod_url = prods[0]["url"]
    with patch("app.scrapers.amazon_scraper.fetch_html", return_value=MOCK_AMAZON_DETAIL_HTML):
        detail_res_str = await get_product_details.ainvoke({"url": top_prod_url})
        detail_res = json.loads(detail_res_str)

    assert detail_res["status"] == "success"
    assert "specs" in detail_res
    print(f"Extracted Specs: {list(detail_res['specs'].keys())[:5]}")

    print("\n=== 3. Testing Comparison Tool ===")
    comp_res_str = await compare_products.ainvoke({"urls": [p["url"] for p in prods[:2]]})
    comp_res = json.loads(comp_res_str)

    assert comp_res["status"] == "success"
    assert "verdict" in comp_res
    print(f"Comparison Verdict: {comp_res['verdict']}")


@pytest.mark.asyncio
async def test_multi_step_agent():
    print("\n=== 4. Testing Autonomous Multi-Step Agent Execution (Mock) ===")
    task = (
        "Find me the best value 32GB DDR5 RAM for a gaming PC under 12000. "
        "Inspect the top candidate details, verify voltage and form factor, "
        "and produce a final structured comparison table."
    )

    # Pre-scripted LLM responses for the 3 steps
    step1_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_amazon",
                "args": {"query": "32GB DDR5 RAM", "max_price": 12000.0},
                "id": "step1_call",
            }
        ],
    )
    step2_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "compare_products",
                "args": {},
                "id": "step2_call",
            }
        ],
    )
    step3_synth = AIMessage(
        content=(
            "# Recommendation: Corsair Vengeance 32GB DDR5 6000MHz CL36\n"
            "| Product | Capacity | Speed | Price | Value Score |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Corsair Vengeance | 32GB | 6000MHz | ₹9,499 | 82.5 |\n\n"
            "This kit is optimal for gaming with tight timings and lifetime warranty."
        ),
        tool_calls=[],
    )

    mock_llm = MockChatGroq([step1_call, step2_call, step3_synth])
    agent = AgentService(llm=mock_llm)

    async def fetch_router(url, timeout=25.0):
        if "s?k=" in url:
            return MOCK_AMAZON_SEARCH_HTML
        return MOCK_AMAZON_DETAIL_HTML

    with patch("app.scrapers.amazon_scraper.fetch_html", side_effect=fetch_router):
        result = await agent.run_task(task, max_steps=5)

    assert len(result.get("steps_taken", [])) >= 1, "Expected at least 1 step taken"
    assert "search_amazon" in (result.get("tools_called") or []), "Expected search_amazon in tools_called"
    assert result.get("final_answer"), "Expected non-empty final answer"
    print("\n>>> Autonomous Multi-Step Agent Test PASSED! <<<\n")


def test_api_endpoints():
    print("=== 4. Testing FastAPI Multi-Step Endpoints (Mock) ===")
    client = TestClient(app)

    step1 = AIMessage(
        content="",
        tool_calls=[{"name": "search_amazon", "args": {"query": "DDR5 RAM"}, "id": "c1"}],
    )
    step2 = AIMessage(
        content="# Final Answer: Corsair Vengeance DDR5 recommended.",
        tool_calls=[],
    )
    mock_llm = MockChatGroq([step1, step2])
    agent = AgentService(llm=mock_llm)

    payload = {"task": "Find me the best DDR5 RAM deal on Amazon"}

    with patch("app.main.agent_service", agent), \
         patch("app.scrapers.amazon_scraper.fetch_html", return_value=MOCK_AMAZON_SEARCH_HTML):
        resp = client.post("/api/research", json=payload)

    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()
    assert len(data.get("products", [])) > 0, "Expected products in research response"
    assert data.get("final_answer"), "Expected final answer in research response"
    print("\n>>> FastAPI Endpoints Test PASSED! <<<\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_tools_directly())
