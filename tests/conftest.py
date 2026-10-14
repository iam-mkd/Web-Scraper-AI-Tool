import os
import sys
import json
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Fixed Product Dataset for Comparison & Ranking Tests
# ---------------------------------------------------------------------------
FIXED_PRODUCTS_DATASET: List[Dict[str, Any]] = [
    {
        "name": "Corsair Vengeance 32GB (2x16GB) DDR5 6000MHz CL36 Desktop RAM",
        "price": 9499.0,
        "capacity_gb": 32,
        "speed_mhz": 6000,
        "cl_latency": 36,
        "rating": 4.5,
        "review_count": 520,
        "form_factor": "UDIMM",
        "kit_size": "2x16GB",
        "voltage": "1.35V",
        "warranty": "Limited Lifetime Warranty",
        "url": "https://www.amazon.in/dp/B0CORSAIR1",
    },
    {
        "name": "Crucial 16GB (1x16GB) DDR5 4800MHz CL40 Desktop RAM",
        "price": 3899.0,
        "capacity_gb": 16,
        "speed_mhz": 4800,
        "cl_latency": 40,
        "rating": 4.2,
        "review_count": 150,
        "form_factor": "UDIMM",
        "kit_size": "1x16GB",
        "voltage": "1.1V",
        "warranty": "10-Year Limited Warranty",
        "url": "https://www.amazon.in/dp/B0CRUCIAL1",
    },
    {
        "name": "G.Skill Trident Z5 RGB 32GB (2x16GB) DDR5 6400MHz CL32 Desktop Gaming Memory",
        "price": 13999.0,
        "capacity_gb": 32,
        "speed_mhz": 6400,
        "cl_latency": 32,
        "rating": 4.8,
        "review_count": 310,
        "form_factor": "UDIMM",
        "kit_size": "2x16GB",
        "voltage": "1.4V",
        "warranty": "Limited Lifetime Warranty",
        "url": "https://www.amazon.in/dp/B0GSKILL01",
    },
    {
        "name": "Crucial 32GB (1x32GB) DDR5 5200MHz SODIMM Laptop Memory",
        "price": 7899.0,
        "capacity_gb": 32,
        "speed_mhz": 5200,
        "cl_latency": 42,
        "rating": 4.6,
        "review_count": 85,
        "form_factor": "SODIMM",
        "kit_size": "1x32GB",
        "voltage": "1.1V",
        "warranty": "10-Year Warranty",
        "url": "https://www.amazon.in/dp/B0CRUCLAP1",
    },
]

UNPRICED_PRODUCT: Dict[str, Any] = {
    "name": "Kingston FURY Beast 32GB (2x16GB) DDR5 5600MHz CL40 Desktop RAM",
    "price": None,
    "capacity_gb": 32,
    "speed_mhz": 5600,
    "cl_latency": 40,
    "rating": 4.3,
    "review_count": 45,
    "form_factor": "UDIMM",
    "kit_size": "2x16GB",
    "voltage": "1.25V",
    "warranty": "Lifetime Warranty",
    "url": "https://www.amazon.in/dp/B0KINGSTON",
}


# ---------------------------------------------------------------------------
# Mock Amazon HTML Pages
# ---------------------------------------------------------------------------
MOCK_AMAZON_SEARCH_HTML = """
<!DOCTYPE html>
<html>
<body>
  <div data-component-type="s-search-result">
    <h2><a href="/Corsair-Vengeance-6000MHz-Desktop-Memory/dp/B0CORSAIR1">Corsair Vengeance 32GB (2x16GB) DDR5 6000MHz CL36 Desktop RAM</a></h2>
    <span class="a-price"><span class="a-offscreen">₹9,499.00</span></span>
    <span class="a-icon-alt">4.5 out of 5 stars</span>
    <span class="a-size-base s-underline-text">520</span>
  </div>

  <div data-component-type="s-search-result">
    <h2><a href="/Crucial-16GB-DDR5-4800MHz-Desktop/dp/B0CRUCIAL1">Crucial 16GB (1x16GB) DDR5 4800MHz CL40 Desktop RAM</a></h2>
    <span class="a-price"><span class="a-offscreen">₹3,899.00</span></span>
    <span class="a-icon-alt">4.2 out of 5 stars</span>
    <span class="a-size-base s-underline-text">150</span>
  </div>

  <div data-component-type="s-search-result">
    <h2><a href="/GSkill-Trident-32GB-6400MHz/dp/B0GSKILL01">G.Skill Trident Z5 RGB 32GB (2x16GB) DDR5 6400MHz CL32 Desktop Gaming Memory</a></h2>
    <span class="a-price"><span class="a-offscreen">₹13,999.00</span></span>
    <span class="a-icon-alt">4.8 out of 5 stars</span>
    <span class="a-size-base s-underline-text">310</span>
  </div>

  <div data-component-type="s-search-result">
    <h2><a href="/Crucial-32GB-SODIMM-Laptop/dp/B0CRUCLAP1">Crucial 32GB (1x32GB) DDR5 5200MHz SODIMM Laptop Memory</a></h2>
    <span class="a-price"><span class="a-offscreen">₹7,899.00</span></span>
    <span class="a-icon-alt">4.6 out of 5 stars</span>
    <span class="a-size-base s-underline-text">85</span>
  </div>

  <div data-component-type="s-search-result">
    <h2><a href="/Kingston-FURY-32GB-DDR5/dp/B0KINGSTON">Kingston FURY Beast 32GB (2x16GB) DDR5 5600MHz CL40 Desktop RAM</a></h2>
    <!-- Out of stock, no price tag -->
    <span class="a-icon-alt">4.3 out of 5 stars</span>
    <span class="a-size-base s-underline-text">45</span>
  </div>
</body>
</html>
"""

MOCK_AMAZON_EMPTY_SEARCH_HTML = """
<!DOCTYPE html>
<html>
<body>
  <div class="s-no-outline">
    <p>No results for <b>nonexistentquery12345xyz</b>.</p>
  </div>
</body>
</html>
"""

MOCK_AMAZON_DETAIL_HTML = """
<!DOCTYPE html>
<html>
<body>
  <h1 id="productTitle">Corsair Vengeance 32GB (2x16GB) DDR5 6000MHz CL36 Desktop RAM</h1>
  <span class="a-price"><span class="a-offscreen">₹9,499.00</span></span>
  <span id="acrPopover"><span class="a-icon-alt">4.5 out of 5 stars</span></span>
  <span id="acrCustomerReviewText">520 ratings</span>

  <div id="feature-bullets">
    <ul>
      <li><span class="a-list-item">Custom Intel XMP 3.0 Profiles for easy overclocking</span></li>
      <li><span class="a-list-item">Solid aluminum heatspreader keeps memory cool</span></li>
      <li><span class="a-list-item">Operating voltage at 1.35V with tight CL36 timings</span></li>
      <li><span class="a-list-item">Limited Lifetime Warranty for total peace of mind</span></li>
    </ul>
  </div>

  <table id="productDetails_techSpec_section_1">
    <tr><th>Brand</th><td>Corsair</td></tr>
    <tr><th>Item model number</th><td>CMK32GX5M2B6000C36</td></tr>
    <tr><th>RAM Memory Technology</th><td>DDR5 6000 MHz</td></tr>
    <tr><th>Form Factor</th><td>UDIMM</td></tr>
    <tr><th>Voltage</th><td>1.35V</td></tr>
    <tr><th>Warranty</th><td>Limited Lifetime Warranty</td></tr>
  </table>
</body>
</html>
"""

MOCK_AMAZON_UNPRICED_DETAIL_HTML = """
<!DOCTYPE html>
<html>
<body>
  <h1 id="productTitle">Kingston FURY Beast 32GB (2x16GB) DDR5 5600MHz CL40 Desktop RAM</h1>
  <span id="acrPopover"><span class="a-icon-alt">4.3 out of 5 stars</span></span>
  <span id="acrCustomerReviewText">45 ratings</span>

  <div id="feature-bullets">
    <ul>
      <li><span class="a-list-item">Currently unavailable. We don't know when or if this item will be back in stock.</span></li>
      <li><span class="a-list-item">Low-profile heat spreader design</span></li>
    </ul>
  </div>

  <table id="productDetails_techSpec_section_1">
    <tr><th>Brand</th><td>Kingston</td></tr>
    <tr><th>Item model number</th><td>KF556C40BBK2-32</td></tr>
    <tr><th>Form Factor</th><td>UDIMM</td></tr>
    <tr><th>Voltage</th><td>1.25V</td></tr>
  </table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Mock LLM for Deterministic Agent Testing
# ---------------------------------------------------------------------------
class MockChatGroq:
    """
    Deterministic Mock LLM for LangChain agent tests.
    Plays back a sequence of pre-scripted AIMessage responses or tool calls.
    """
    def __init__(self, responses: Optional[List[AIMessage]] = None):
        self.responses = list(responses or [])
        self.call_history: List[Any] = []
        self.bound_tools: List[Any] = []

    def bind_tools(self, tools: List[Any]) -> "MockChatGroq":
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages: List[Any]) -> AIMessage:
        self.call_history.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return AIMessage(content="Default mock response: Task completed successfully.", tool_calls=[])


@pytest.fixture
def fixed_products():
    return list(FIXED_PRODUCTS_DATASET)


@pytest.fixture
def unpriced_product():
    return dict(UNPRICED_PRODUCT)


@pytest.fixture
def mock_amazon_search_html():
    return MOCK_AMAZON_SEARCH_HTML


@pytest.fixture
def mock_amazon_detail_html():
    return MOCK_AMAZON_DETAIL_HTML


@pytest.fixture
def mock_amazon_unpriced_detail_html():
    return MOCK_AMAZON_UNPRICED_DETAIL_HTML
