import pytest
import json
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage

from app.scrapers.base_scraper import fetch_html
from app.scrapers.amazon_scraper import search_amazon_products, get_amazon_product_details
from app.agent.service import (
    search_amazon,
    scrape_url,
    calculate_value_score,
    build_product_comparison,
    extract_retry_after,
    AgentService,
)
from app.models import Product
from tests.conftest import MockChatGroq, FIXED_PRODUCTS_DATASET


# ===========================================================================
# 1. Amazon returns 503 -> Bounded retries, then graceful failure
# ===========================================================================
@pytest.mark.asyncio
async def test_amazon_503_bounded_retries_and_graceful_failure():
    """
    Simulate Amazon returning HTTP 503 Service Unavailable.
    Assert bounded retries occur (exactly 3 attempts with sleep),
    and the search_amazon tool reports a graceful failure JSON rather than crashing.
    """
    attempt_count = 0

    async def mock_fetch_503(url, timeout=25.0):
        nonlocal attempt_count
        attempt_count += 1
        request = httpx.Request("GET", url)
        response = httpx.Response(503, request=request)
        raise httpx.HTTPStatusError("503 Service Unavailable", request=request, response=response)

    with patch("app.scrapers.amazon_scraper.fetch_html", side_effect=mock_fetch_503), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        
        # Test 1: search_amazon_products executes bounded retries
        with pytest.raises(httpx.HTTPStatusError):
            await search_amazon_products("DDR5 RAM", max_attempts=3)
        assert attempt_count == 3
        assert mock_sleep.call_count == 2  # slept between attempts 1->2 and 2->3

        # Test 2: search_amazon tool catches bounded failure gracefully
        attempt_count = 0
        tool_res_str = await search_amazon.ainvoke({"query": "DDR5 RAM"})
        tool_res = json.loads(tool_res_str)

        assert tool_res["status"] == "error"
        assert "503" in tool_res["message"] or "unavailable" in tool_res["message"].lower()
        assert tool_res["products"] == []
        assert tool_res["stats"]["total_scraped"] == 0


# ===========================================================================
# 2. Groq returns 429 -> Respect retry guidance; bounded backoff
# ===========================================================================
def test_extract_retry_after_from_headers_and_message():
    """Verify that retry guidance is extracted from headers, error messages, and bounded."""
    # A. From HTTP response headers
    req = httpx.Request("POST", "https://api.groq.com")
    resp = httpx.Response(429, headers={"retry-after": "3.5"}, request=req)
    err_with_header = httpx.HTTPStatusError("Rate limited", request=req, response=resp)
    assert extract_retry_after(err_with_header) == 3.5

    # B. From error message text regex
    err_with_msg = Exception("Groq TPM limit reached. Please try again in 2.8s.")
    assert extract_retry_after(err_with_msg) == 2.8

    # C. Bounded backoff caps large values
    err_huge = Exception("Please try again in 60s.")
    assert extract_retry_after(err_huge, max_backoff=10.0) == 10.0


@pytest.mark.asyncio
async def test_groq_429_bounded_backoff_recovery():
    """
    Simulate Groq returning 429 rate limit on attempt 1 with retry guidance,
    then succeeding on attempt 2.
    Assert bounded backoff is respected and execution finishes cleanly.
    """
    fail_429 = Exception("Groq 429 Too Many Requests: Rate limit reached. Please try again in 1.5s.")
    success_msg = AIMessage(content="Here are the search results.", tool_calls=[])

    mock_llm = MagicMock()
    # First invocation fails with 429, second invocation succeeds
    mock_llm.ainvoke = AsyncMock(side_effect=[fail_429, success_msg])
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    agent = AgentService(llm=mock_llm)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await agent.run_task("Recommend RAM", max_steps=3)

    assert mock_sleep.call_count == 1
    # Slept the extracted retry guidance duration (1.5s)
    mock_sleep.assert_called_with(1.5)
    assert result["final_answer"] == "Here are the search results."


# ===========================================================================
# 3. Request times out -> Stop waiting and report a useful error
# ===========================================================================
@pytest.mark.asyncio
async def test_request_timeout_reports_useful_error():
    """Simulate network timeout and assert useful error without indefinite hanging."""
    async def mock_timeout(url, timeout=25.0):
        raise TimeoutError(f"Request to '{url}' timed out after {timeout}s.")

    # 1. Test scraper tool handling
    with patch("app.scrapers.base_scraper.fetch_html", side_effect=mock_timeout):
        scrape_res_str = await scrape_url.ainvoke({"url": "https://slow-server.com"})
        scrape_res = json.loads(scrape_res_str)
        assert scrape_res["status"] == "error"
        assert "timed out" in scrape_res["message"].lower()

    # 2. Test Amazon search tool handling
    with patch("app.scrapers.amazon_scraper.fetch_html", side_effect=mock_timeout), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        amazon_res_str = await search_amazon.ainvoke({"query": "DDR5 RAM"})
        amazon_res = json.loads(amazon_res_str)
        assert amazon_res["status"] == "error"
        assert "timed out" in amazon_res["message"].lower() or "failed" in amazon_res["message"].lower()


# ===========================================================================
# 4. Product has no price -> Don't invent one or crash
# ===========================================================================
def test_product_no_price_does_not_crash_or_invent():
    """
    Assert that unpriced products:
    1. Do NOT fail Pydantic model validation.
    2. Do NOT invent prices like 10000.0 or 0.0.
    3. Return None for Value Score.
    4. Are NEVER chosen as budget_pick.
    """
    # 1. Model validation accepts None for price
    prod = Product(
        name="Kingston FURY Beast 32GB DDR5 5600MHz",
        price=None,
        capacity_gb=32,
        url="https://www.amazon.in/dp/B0KINGSTON",
    )
    assert prod.price is None

    # 2. calculate_value_score returns None without inventing price
    val_score = calculate_value_score(
        capacity_gb=32,
        speed_mhz=5600,
        cl_latency=40,
        rating=4.5,
        review_count=100,
        price=None,
    )
    assert val_score is None

    # 3. build_product_comparison handles unpriced item safely
    priced_item = {
        "name": "Crucial 16GB RAM",
        "price": 3899.0,
        "capacity_gb": 16,
        "speed_mhz": 4800,
        "cl_latency": 40,
        "url": "https://www.amazon.in/dp/B0CRUCIAL1",
    }
    unpriced_item = {
        "name": "Kingston Unpriced RAM",
        "price": None,
        "capacity_gb": 32,
        "speed_mhz": 5600,
        "cl_latency": 40,
        "url": "https://www.amazon.in/dp/B0KINGSTON",
    }

    comp = build_product_comparison([priced_item, unpriced_item])
    assert comp["status"] == "success"
    # Budget pick must be the priced item, never the unpriced item
    assert comp["verdict"]["budget_pick"]["price"] == 3899.0
    assert comp["verdict"]["budget_pick"]["name"] == "Crucial 16GB RAM"

    # Unpriced item has cons note
    comp_unpriced = next(x for x in comp["comparison_table"] if x["name"] == "Kingston Unpriced RAM")
    assert comp_unpriced["price"] is None
    assert comp_unpriced["price_per_gb"] is None
    assert any("unavailable" in c.lower() or "price" in c.lower() for c in comp_unpriced["cons"])


# ===========================================================================
# 5. Agent repeats a tool call -> Detect repetition or stop at step limit
# ===========================================================================
@pytest.mark.asyncio
async def test_agent_repeats_tool_call_detection_and_step_limit():
    """
    Simulate LLM emitting the identical tool call repeatedly.
    Assert that the agent detects repetition, avoids redundant re-execution,
    and terminates strictly at the step limit (max_steps).
    """
    repeated_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_amazon",
                "args": {"query": "DDR5 RAM"},
                "id": "repeated_call_id",
            }
        ],
    )

    # LLM keeps returning the same search_amazon tool call every step
    mock_llm = MockChatGroq([repeated_call, repeated_call, repeated_call, repeated_call])
    agent = AgentService(llm=mock_llm)

    sample_products = [
        {
            "name": "Crucial 16GB RAM",
            "price": 3899.0,
            "capacity_gb": 16,
            "speed_mhz": 4800,
            "cl_latency": 40,
            "rating": 4.2,
            "review_count": 100,
            "form_factor": "UDIMM",
            "kit_size": "1x16GB",
            "url": "https://www.amazon.in/dp/B0CRUCIAL1",
        }
    ]

    with patch("app.agent.service.search_amazon_products", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = sample_products
        result = await agent.run_task("Find RAM", max_steps=3)

    # Assert underlying scraper function was only executed ONCE, avoiding duplicate queries!
    assert mock_search.call_count == 1
    # Assert repetition was recorded in step history
    repetition_flags = [s.get("repetition_detected") for s in result["steps_taken"]]
    assert any(repetition_flags)
    # Assert execution stopped strictly at step limit (max_steps=3)
    assert len(result["steps_taken"]) <= 3


# ===========================================================================
# 6. No products match -> Explain that no suitable results were found
# ===========================================================================
@pytest.mark.asyncio
async def test_no_products_match_explanation():
    """
    Simulate search where 0 products match budget/specifications.
    Assert that search_amazon tool returns not_found status,
    and the agent explains clearly that no suitable results were found.
    """
    # 1. Tool test: search_amazon with extreme filter
    with patch("app.scrapers.amazon_scraper.search_amazon_products", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        tool_res_str = await search_amazon.ainvoke({"query": "DDR5 RAM", "max_price": 500.0})
        tool_res = json.loads(tool_res_str)

        assert tool_res["status"] == "not_found"
        assert "no products found" in tool_res["message"].lower()

    # 2. Agent test: empty products fallback explanation
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "search_amazon", "args": {"query": "impossible query"}, "id": "call_1"}],
    )
    # LLM emits tool call on step 1, and on step 2 produces empty content so agent triggers fallback synthesis
    empty_content = AIMessage(content="", tool_calls=[])
    mock_llm = MockChatGroq([tool_call, empty_content])
    agent = AgentService(llm=mock_llm)

    with patch("app.scrapers.amazon_scraper.search_amazon_products", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        result = await agent.run_task("Find RAM for 100 rupees", max_steps=2)

    assert result["products"] == []
    assert "no suitable products" in result["final_answer"].lower() or "found" in result["final_answer"].lower()
