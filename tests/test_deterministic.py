import pytest
import json
from unittest.mock import patch
from langchain_core.messages import AIMessage

from app.agent.service import (
    calculate_value_score,
    build_product_comparison,
    search_amazon,
    AgentService,
)
from app.scrapers.amazon_scraper import search_amazon_products, get_amazon_product_details
from tests.conftest import MockChatGroq, FIXED_PRODUCTS_DATASET, UNPRICED_PRODUCT


# ===========================================================================
# 1. Deterministic Value Score & Comparison Matrix Tests
# ===========================================================================
def test_calculate_value_score_deterministic(fixed_products):
    """Assert value scores are computed deterministically within [0, 100]."""
    for p in fixed_products:
        score = calculate_value_score(
            capacity_gb=p["capacity_gb"],
            speed_mhz=p["speed_mhz"],
            cl_latency=p["cl_latency"],
            rating=p["rating"],
            review_count=p["review_count"],
            price=p["price"],
        )
        assert score is not None
        assert 0.0 <= score <= 100.0

    # Corsair (6000MHz CL36, Rs. 9,499) vs Crucial (4800MHz CL40, Rs. 3,899)
    corsair = fixed_products[0]
    crucial = fixed_products[1]
    score_corsair = calculate_value_score(
        capacity_gb=corsair["capacity_gb"],
        speed_mhz=corsair["speed_mhz"],
        cl_latency=corsair["cl_latency"],
        rating=corsair["rating"],
        review_count=corsair["review_count"],
        price=corsair["price"],
    )
    score_crucial = calculate_value_score(
        capacity_gb=crucial["capacity_gb"],
        speed_mhz=crucial["speed_mhz"],
        cl_latency=crucial["cl_latency"],
        rating=crucial["rating"],
        review_count=crucial["review_count"],
        price=crucial["price"],
    )
    assert score_corsair > 0.0
    assert score_crucial > 0.0


def test_calculate_value_score_unpriced(unpriced_product):
    """Unpriced products should return None without crashing or inventing a score."""
    score = calculate_value_score(
        capacity_gb=unpriced_product["capacity_gb"],
        speed_mhz=unpriced_product["speed_mhz"],
        cl_latency=unpriced_product["cl_latency"],
        rating=unpriced_product["rating"],
        review_count=unpriced_product["review_count"],
        price=unpriced_product["price"],
    )
    assert score is None


def test_build_product_comparison_verdicts(fixed_products, unpriced_product):
    """
    Test deterministic comparison matrix and verdict crowns:
    - best_performance: highest clock speed and tightest CAS latency (G.Skill 6400MHz CL32)
    - budget_pick: lowest positive price (Crucial Rs. 3,899)
    - unpriced item is NOT picked as budget pick
    """
    all_items = fixed_products + [unpriced_product]
    comparison = build_product_comparison(all_items)

    assert comparison["status"] == "success"
    assert comparison["products_compared"] == len(all_items)

    verdict = comparison["verdict"]
    # 1. Best Performance should be G.Skill Trident (6400MHz, CL32)
    assert "G.Skill" in verdict["best_performance"]["name"]
    assert verdict["best_performance"]["speed_mhz"] == 6400

    # 2. Budget pick should be Crucial (Rs. 3,899), NOT the unpriced Kingston product
    assert verdict["budget_pick"]["price"] == 3899.0
    assert "Crucial" in verdict["budget_pick"]["name"]
    assert verdict["budget_pick"]["name"] != unpriced_product["name"]

    # 3. Best value should be crowned with valid value score
    assert verdict["best_value"]["value_score"] is not None
    assert verdict["best_value"]["value_score"] > 0


# ===========================================================================
# 2. Mock Amazon HTML Parser Tests
# ===========================================================================
@pytest.mark.asyncio
async def test_amazon_search_parser_mock(mock_amazon_search_html):
    """Test parsing search results from mock Amazon HTML without network calls."""
    with patch("app.scrapers.amazon_scraper.fetch_html", return_value=mock_amazon_search_html):
        products = await search_amazon_products("DDR5 RAM")

    assert len(products) >= 4
    names = [p["name"] for p in products]
    assert any("Corsair" in n for n in names)
    assert any("Crucial" in n for n in names)
    assert any("G.Skill" in n for n in names)

    # First item price parsing
    corsair = next(p for p in products if "Corsair" in p["name"])
    assert corsair["price"] == 9499.0
    assert corsair["capacity_gb"] == 32
    assert corsair["speed_mhz"] == 6000
    assert corsair["cl_latency"] == 36
    assert corsair["rating"] == 4.5
    assert corsair["form_factor"] == "UDIMM"

    # Laptop RAM form factor parsing
    sodimm = next(p for p in products if "SODIMM" in p["name"])
    assert sodimm["form_factor"] == "SODIMM"

    # Unpriced item parsed safely with price=None
    kingston = next(p for p in products if "Kingston" in p["name"])
    assert kingston["price"] is None


@pytest.mark.asyncio
async def test_amazon_detail_parser_mock(mock_amazon_detail_html):
    """Test parsing granular specs from mock Amazon product detail page."""
    with patch("app.scrapers.amazon_scraper.fetch_html", return_value=mock_amazon_detail_html):
        details = await get_amazon_product_details("https://www.amazon.in/dp/B0CORSAIR1")

    assert details["name"] == "Corsair Vengeance 32GB (2x16GB) DDR5 6000MHz CL36 Desktop RAM"
    assert details["price"] == 9499.0
    assert details["voltage"] == "1.35V"
    assert details["brand"] == "Corsair"
    assert details["form_factor"] == "UDIMM"
    assert details["warranty"] == "Limited Lifetime Warranty"
    assert len(details["highlights"]) > 0


# ===========================================================================
# 3. Deterministic Tool Selection & Agent Loop with Mock Groq
# ===========================================================================
@pytest.mark.asyncio
async def test_agent_tool_selection_mock(mock_amazon_search_html):
    """
    Assert that the agent selects search_amazon tool and executes deterministically
    using a mocked ChatGroq response.
    """
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_amazon",
                "args": {"query": "DDR5 RAM", "max_price": 50000.0, "form_factor": "UDIMM"},
                "id": "call_mock_1",
            }
        ],
    )
    final_synthesis_response = AIMessage(
        content="Here is the best deal: Corsair Vengeance DDR5 at Rs. 9,499.",
        tool_calls=[],
    )

    mock_llm = MockChatGroq([tool_call_response, final_synthesis_response])
    agent = AgentService(llm=mock_llm)

    with patch("app.scrapers.amazon_scraper.fetch_html", return_value=mock_amazon_search_html):
        result = await agent.run_task("Find me DDR5 RAM deals on Amazon under 50000", max_steps=3)

    assert result["tool_called"] == "search_amazon"
    assert len(result["tools_called"]) == 1
    assert len(result["products"]) > 0
    assert "cheapest_product" in (result["stats"] or {})
    assert "Corsair" in result["final_answer"]
    assert len(result["steps_taken"]) == 1
    assert result["steps_taken"][0]["tool"] == "search_amazon"


@pytest.mark.asyncio
async def test_agent_multi_step_loop_mock(mock_amazon_search_html, mock_amazon_detail_html):
    """
    Test multi-step autonomous loop:
    Step 1: search_amazon -> Step 2: get_product_details -> Step 3: compare_products -> Final Answer
    All executed offline with Mock LLM and Mock Amazon HTML.
    """
    step1_search = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_amazon",
                "args": {"query": "DDR5 RAM 32GB", "max_price": 15000.0},
                "id": "step1_call",
            }
        ],
    )
    step2_details = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_product_details",
                "args": {"url": "https://www.amazon.in/dp/B0CORSAIR1"},
                "id": "step2_call",
            }
        ],
    )
    step3_compare = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "compare_products",
                "args": {"product_urls": ["https://www.amazon.in/dp/B0CORSAIR1"]},
                "id": "step3_call",
            }
        ],
    )
    step4_final = AIMessage(
        content="# Recommendation\nCorsair Vengeance wins with Value Score 82.5/100.\n| Product | Price |\n|---|---|",
        tool_calls=[],
    )

    mock_llm = MockChatGroq([step1_search, step2_details, step3_compare, step4_final])
    agent = AgentService(llm=mock_llm)

    with patch("app.scrapers.amazon_scraper.fetch_html") as mock_fetch:
        def fetch_router(url, timeout=25.0):
            if "dp" in url:
                return mock_amazon_detail_html
            return mock_amazon_search_html

        mock_fetch.side_effect = fetch_router
        result = await agent.run_task("Find me the best 32GB DDR5 RAM", max_steps=5)

    assert result["tools_called"] == ["search_amazon", "get_product_details", "compare_products"]
    assert len(result["steps_taken"]) == 3
    assert len(result["detailed_products"]) == 1
    assert result["comparison"] is not None
    assert "Corsair" in result["final_answer"]
