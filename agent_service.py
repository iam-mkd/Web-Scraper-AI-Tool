import os
import json
import math
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

from structured_scraper import search_amazon_products, get_amazon_product_details
from scraper import scrape_webpage
from vector_store import get_vector_store_manager, VectorStoreManager
from models import Product, ProductDetail


# ---------------------------------------------------------------------------
# Global Session Product Cache & Helper
# ---------------------------------------------------------------------------
_PRODUCTS_CACHE: Dict[str, Dict[str, Any]] = {}


def register_product_in_cache(product: Dict[str, Any]) -> None:
    """Stores and enriches product data in the session cache."""
    url = product.get("url")
    if url:
        if url in _PRODUCTS_CACHE:
            _PRODUCTS_CACHE[url].update(product)
        else:
            _PRODUCTS_CACHE[url] = dict(product)


# ---------------------------------------------------------------------------
# Deterministic Value Score & Comparison Engine
# ---------------------------------------------------------------------------
def calculate_value_score(
    capacity_gb: int,
    speed_mhz: Optional[int],
    cl_latency: Optional[int],
    rating: Optional[float],
    review_count: Optional[int],
    price: float,
) -> float:
    """
    Computes a deterministic Value Score (0.0 - 100.0) based on:
    Value Score = Performance (MHz + CL) + Capacity + Rating + Review Confidence - Price Penalty (INR/GB)
    """
    # 1. Capacity Factor (max 30 pts):
    cap_score = min(30.0, (capacity_gb / 32.0) * 28.0)

    # 2. Performance Factor (max 25 pts):
    speed = speed_mhz or 4800
    speed_score = min(15.0, max(0.0, (speed - 4800.0) / 1600.0 * 15.0))
    cl = cl_latency or 40
    cl_score = min(10.0, max(0.0, (46.0 - cl) / 16.0 * 10.0))
    perf_score = speed_score + cl_score

    # 3. Rating Factor (max 15 pts):
    r = rating if rating is not None else 4.0
    rating_score = (r / 5.0) * 15.0

    # 4. Review Confidence (max 10 pts):
    revs = review_count if review_count is not None else 5
    conf_score = min(10.0, (math.log10(max(revs, 1)) / 3.0) * 10.0)

    # 5. Price Efficiency Factor (max 20 pts):
    price_per_gb = price / max(capacity_gb, 1)
    price_eff_score = min(20.0, max(0.0, (2500.0 - price_per_gb) / 1300.0 * 20.0))

    total = cap_score + perf_score + rating_score + conf_score + price_eff_score
    return round(max(0.0, min(100.0, total)), 1)


def build_product_comparison(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministically compiles a side-by-side comparison matrix with pros, cons,
    and clear verdicts (best_value, best_performance, budget_pick).
    """
    comparison_items: List[Dict[str, Any]] = []

    for p in products:
        name = p.get("name", "Unknown Product")
        price = p.get("price") or 0.0
        cap = p.get("capacity_gb") or 16
        speed = p.get("speed_mhz")
        cl = p.get("cl_latency")
        voltage = p.get("voltage") or "1.25V - 1.35V standard"
        warranty = p.get("warranty") or "Manufacturer Warranty"
        form_factor = p.get("form_factor") or "UDIMM"
        kit_size = p.get("kit_size") or f"1x{cap}GB"

        val_score = p.get("value_score")
        if val_score is None:
            val_score = calculate_value_score(
                capacity_gb=cap,
                speed_mhz=speed,
                cl_latency=cl,
                rating=p.get("rating"),
                review_count=p.get("review_count"),
                price=price,
            )

        price_per_gb = round(price / max(cap, 1), 1)

        pros: List[str] = []
        cons: List[str] = []

        # Performance assessment
        if speed and speed >= 6000:
            pros.append(f"High clock speed ({speed}MHz) ideal for modern gaming & bandwidth")
        elif speed and speed <= 5200:
            cons.append(f"Entry-level speed ({speed}MHz)")

        # Latency assessment
        if cl and cl <= 32:
            pros.append(f"Tight CAS latency (CL{cl}) with fast responsiveness")
        elif cl and cl >= 40:
            cons.append(f"Looser CAS latency (CL{cl})")

        # Capacity assessment
        if cap >= 32:
            pros.append(f"Generous {cap}GB total capacity ({kit_size}) for heavy multitasking")
        elif cap <= 8:
            cons.append("Limited 8GB capacity for modern DDR5 workloads")

        # Price efficiency
        if price_per_gb < 1500:
            pros.append(f"High price efficiency (₹{price_per_gb}/GB)")
        elif price_per_gb > 2000:
            cons.append(f"Premium cost per GB (₹{price_per_gb}/GB)")

        if val_score >= 70:
            pros.append(f"Outstanding Value Score ({val_score}/100)")

        comparison_items.append({
            "name": name,
            "url": p.get("url"),
            "price": price,
            "capacity_gb": cap,
            "kit_size": kit_size,
            "price_per_gb": price_per_gb,
            "speed_mhz": speed,
            "cl_latency": cl,
            "voltage": voltage,
            "form_factor": form_factor,
            "warranty": warranty,
            "value_score": val_score,
            "rating": p.get("rating"),
            "review_count": p.get("review_count"),
            "pros": pros,
            "cons": cons,
        })

    best_value = max(comparison_items, key=lambda x: x["value_score"]) if comparison_items else None
    best_perf = max(comparison_items, key=lambda x: (x.get("speed_mhz") or 0, -(x.get("cl_latency") or 99))) if comparison_items else None
    budget_pick = min(comparison_items, key=lambda x: x["price"]) if comparison_items else None

    return {
        "status": "success",
        "products_compared": len(comparison_items),
        "comparison_table": comparison_items,
        "verdict": {
            "best_value": {
                "name": best_value["name"] if best_value else None,
                "value_score": best_value["value_score"] if best_value else None,
                "price": best_value["price"] if best_value else None,
                "reason": "Highest overall value score balancing capacity, speed, latency, and cost per GB",
            },
            "best_performance": {
                "name": best_perf["name"] if best_perf else None,
                "speed_mhz": best_perf["speed_mhz"] if best_perf else None,
                "cl_latency": best_perf["cl_latency"] if best_perf else None,
                "price": best_perf["price"] if best_perf else None,
                "reason": "Top speed and lowest CAS latency among compared items",
            },
            "budget_pick": {
                "name": budget_pick["name"] if budget_pick else None,
                "price": budget_pick["price"] if budget_pick else None,
                "capacity_gb": budget_pick["capacity_gb"] if budget_pick else None,
                "reason": "Lowest initial purchase cost",
            },
        },
    }



# ---------------------------------------------------------------------------
# Tool 1: search_amazon (Amazon India Product Search + Deterministic Processing)
# ---------------------------------------------------------------------------
@tool
async def search_amazon(
    query: str,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    capacity_gb: Optional[int] = None,
    speed_mhz: Optional[int] = None,
    cl_rating: Optional[int] = None,
    form_factor: Optional[str] = None,
    kit_size: Optional[str] = None,
    optimization: Optional[str] = "best_value",
    sort_by: Optional[str] = "value_desc",
) -> str:
    """
    Search Amazon India (amazon.in) for products by keyword with optional budget (max_price), rating, speed, capacity, and form factor filters.
    Computes deterministic Value Scores (0-100) taking into account Performance (MHz & CL), Capacity, Rating, Reviews, and Price-per-GB.
    form_factor: 'UDIMM' (Desktop) or 'SODIMM' (Laptop).
    cl_rating: max CAS latency (e.g. 30, 36).
    optimization: 'best_value', 'cheapest', 'performance', 'rating'.
    sort_by: 'value_desc', 'price_asc', 'price_desc', 'rating_desc', 'reviews_desc'.
    """
    products = await search_amazon_products(query)

    if not products:
        return json.dumps({
            "status": "not_found",
            "message": f"No products found on Amazon India for query: '{query}'.",
            "products": [],
            "stats": {},
        })

    # Compute deterministic Value Score and price-per-GB for each product
    for p in products:
        cap = p.get("capacity_gb") or 16
        prc = p.get("price") or 10000.0
        p["value_score"] = calculate_value_score(
            capacity_gb=cap,
            speed_mhz=p.get("speed_mhz"),
            cl_latency=p.get("cl_latency"),
            rating=p.get("rating"),
            review_count=p.get("review_count"),
            price=prc,
        )
        p["price_per_gb"] = round(prc / max(cap, 1), 1)

    # Deterministic Python filtering
    filtered = products

    if max_price is not None:
        filtered = [p for p in filtered if p.get("price") is not None and p["price"] <= max_price]

    if min_rating is not None:
        filtered = [p for p in filtered if p.get("rating") is not None and p["rating"] >= min_rating]

    if capacity_gb is not None:
        filtered = [p for p in filtered if p.get("capacity_gb") == capacity_gb]

    if speed_mhz is not None:
        filtered = [p for p in filtered if p.get("speed_mhz") == speed_mhz]

    if cl_rating is not None:
        filtered = [p for p in filtered if p.get("cl_latency") is not None and p["cl_latency"] <= cl_rating]

    if form_factor is not None:
        filtered = [p for p in filtered if p.get("form_factor", "").upper() == form_factor.upper()]

    if kit_size is not None:
        filtered = [p for p in filtered if p.get("kit_size") == kit_size]

    # Deterministic Python sorting
    if optimization == "cheapest" or sort_by == "price_asc":
        filtered.sort(key=lambda p: (p.get("price") if p.get("price") is not None else float("inf")))
    elif sort_by == "price_desc":
        filtered.sort(key=lambda p: (p.get("price") if p.get("price") is not None else -1), reverse=True)
    elif optimization == "performance":
        filtered.sort(key=lambda p: (p.get("speed_mhz") or 0, -(p.get("cl_latency") or 99)), reverse=True)
    elif optimization == "rating" or sort_by == "rating_desc":
        filtered.sort(key=lambda p: (p.get("rating") or 0), reverse=True)
    elif sort_by == "reviews_desc":
        filtered.sort(key=lambda p: (p.get("review_count") or 0), reverse=True)
    else:
        # Default optimization == "best_value" or sort_by == "value_desc"
        filtered.sort(key=lambda p: (p.get("value_score") or 0), reverse=True)

    # Deterministic summary calculations
    stats: Dict[str, Any] = {
        "total_scraped": len(products),
        "matching_criteria": len(filtered),
    }

    if filtered:
        # 1. Best Value Product (highest Value Score)
        best_value = max(filtered, key=lambda p: (p.get("value_score") or 0))
        stats["best_value_product"] = {
            "name": best_value["name"],
            "value_score": best_value["value_score"],
            "capacity_gb": best_value.get("capacity_gb"),
            "speed_mhz": best_value.get("speed_mhz"),
            "cl_latency": best_value.get("cl_latency"),
            "form_factor": best_value.get("form_factor"),
            "price": best_value["price"],
            "price_per_gb": best_value.get("price_per_gb"),
            "url": best_value["url"],
        }

        # 2. Cheapest Product
        cheapest = min(filtered, key=lambda p: p["price"])
        stats["cheapest_product"] = {
            "name": cheapest["name"],
            "price": cheapest["price"],
            "capacity_gb": cheapest.get("capacity_gb"),
            "form_factor": cheapest.get("form_factor"),
            "url": cheapest["url"],
        }

        # 3. Highest Performance Product
        highest_perf = max(filtered, key=lambda p: (p.get("speed_mhz") or 0, -(p.get("cl_latency") or 99)))
        stats["highest_performance_product"] = {
            "name": highest_perf["name"],
            "speed_mhz": highest_perf.get("speed_mhz"),
            "cl_latency": highest_perf.get("cl_latency"),
            "price": highest_perf["price"],
        }

        # 4. Highest Rated Product
        highest_rated = max(filtered, key=lambda p: (p.get("rating") or 0))
        stats["highest_rated_product"] = {
            "name": highest_rated["name"],
            "rating": highest_rated.get("rating"),
            "review_count": highest_rated.get("review_count"),
            "price": highest_rated["price"],
        }

        # 5. Form Factor Breakdown
        udimm_count = sum(1 for p in filtered if p.get("form_factor") == "UDIMM")
        sodimm_count = sum(1 for p in filtered if p.get("form_factor") == "SODIMM")
        stats["form_factor_breakdown"] = {
            "UDIMM_desktop": udimm_count,
            "SODIMM_laptop": sodimm_count,
        }

        # 6. Price Range
        prices = [p["price"] for p in filtered]
        stats["price_range"] = {
            "min_price": min(prices),
            "max_price": max(prices),
            "avg_price": round(sum(prices) / len(prices), 2),
        }

    # Store in session product cache for multi-step retrieval and comparison
    for p in filtered:
        register_product_in_cache(p)

    # Compact product list to conserve Groq tokens per minute (TPM)
    compact_prods = [
        {
            "name": p.get("name")[:75],
            "price": p.get("price"),
            "capacity_gb": p.get("capacity_gb"),
            "speed_mhz": p.get("speed_mhz"),
            "cl_latency": p.get("cl_latency"),
            "form_factor": p.get("form_factor"),
            "kit_size": p.get("kit_size"),
            "value_score": p.get("value_score"),
            "price_per_gb": p.get("price_per_gb"),
            "url": p.get("url"),
        }
        for p in filtered[:5]
    ]

    payload = {
        "query": query,
        "total_scraped": stats.get("total_scraped"),
        "matching_criteria": stats.get("matching_criteria"),
        "best_value": stats.get("best_value_product"),
        "cheapest": stats.get("cheapest_product"),
        "products": compact_prods,
    }

    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Tool 2: get_product_details (Amazon India Granular Technical Specs)
# ---------------------------------------------------------------------------
@tool
async def get_product_details(url: str) -> str:
    """
    Fetches deep, granular technical specifications for an Amazon India product by URL.
    Extracts operating voltage (e.g. 1.25V, 1.35V), manufacturer warranty, form factor (UDIMM/SODIMM),
    speed (MHz), sub-timings/CAS latency, exact model number, and top feature highlights.
    """
    try:
        details = await get_amazon_product_details(url)
        register_product_in_cache(details)
        compact_details = {
            "name": details.get("name", "Unknown")[:75],
            "url": details.get("url"),
            "price": details.get("price"),
            "voltage": details.get("voltage"),
            "warranty": details.get("warranty"),
            "form_factor": details.get("form_factor"),
            "speed_mhz": details.get("speed_mhz"),
            "cl_latency": details.get("cl_latency"),
            "capacity_gb": details.get("capacity_gb"),
            "kit_size": details.get("kit_size"),
            "brand": details.get("brand"),
            "model_number": details.get("model_number"),
            "highlights": details.get("highlights", [])[:2],
        }
        return json.dumps(compact_details, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "url": url, "message": str(e)})



# ---------------------------------------------------------------------------
# Tool 3: compare_products (Deterministic Side-by-Side Product Comparison)
# ---------------------------------------------------------------------------
@tool
async def compare_products(product_urls: Optional[List[str]] = None) -> str:
    """
    Compares 2 to 4 products side-by-side deterministically based on hardware specs,
    price, capacity, speed, latency, voltage, value score, and warranty.
    Generates structured comparison metrics, pros/cons, and category verdicts (best_value, best_performance, budget_pick).
    """
    products_to_compare: List[Dict[str, Any]] = []

    if product_urls:
        for u in product_urls:
            if u in _PRODUCTS_CACHE:
                products_to_compare.append(_PRODUCTS_CACHE[u])
            else:
                try:
                    d = await get_amazon_product_details(u)
                    register_product_in_cache(d)
                    products_to_compare.append(d)
                except Exception:
                    pass

    # If no URLs provided or none found in cache, use top candidates currently in cache
    if not products_to_compare and _PRODUCTS_CACHE:
        products_to_compare = list(_PRODUCTS_CACHE.values())[:4]

    if not products_to_compare:
        return json.dumps({
            "status": "error",
            "message": "No products found to compare. Please run search_amazon first.",
        })

    comparison = build_product_comparison(products_to_compare)
    return json.dumps(comparison, indent=2)


# Alias tool for backwards compatibility
@tool
async def search_web(
    query: str,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    capacity_gb: Optional[int] = None,
    sort_by: Optional[str] = "price_asc",
) -> str:
    """
    Search Amazon India (amazon.in) for products by keyword and budget.
    """
    return await search_amazon.ainvoke({
        "query": query,
        "max_price": max_price,
        "min_rating": min_rating,
        "capacity_gb": capacity_gb,
        "sort_by": sort_by,
    })


# ---------------------------------------------------------------------------
# Tool 4: scrape_url (General Webpage Scraper & Cleaner)
# ---------------------------------------------------------------------------
@tool
async def scrape_url(url: str) -> str:
    """Scrapes a general webpage URL and returns its cleaned title and text content."""
    try:
        data = await scrape_webpage(url)
        return json.dumps({
            "status": "success",
            "url": data["url"],
            "title": data["title"],
            "content_preview": data["content"][:2000],
            "char_count": data["char_count"],
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ---------------------------------------------------------------------------
# Tool 5: search_rag (ChromaDB Vector Retrieval)
# ---------------------------------------------------------------------------
@tool
async def search_rag(url: str, query: str) -> str:
    """Searches the persistent ChromaDB vector store for indexed content belonging to the given URL."""
    mgr = get_vector_store_manager()
    if not mgr.is_url_indexed(url):
        return json.dumps({
            "status": "not_indexed",
            "message": f"URL '{url}' is not indexed in ChromaDB yet.",
        })
    try:
        results = await mgr.vector_store.asimilarity_search_with_relevance_scores(
            query=query,
            k=4,
            filter={"url": str(url)},
        )
        chunks = [{"content": doc.page_content, "score": round(float(score), 4)} for doc, score in results]
        return json.dumps({"status": "success", "url": url, "chunks": chunks})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ---------------------------------------------------------------------------
# Tool 6: extract_information (Information Extractor)
# ---------------------------------------------------------------------------
@tool
def extract_information(text: str, schema_description: str) -> str:
    """Extracts specific target attributes from a given block of text based on schema guidelines."""
    return json.dumps({
        "status": "success",
        "guideline": schema_description,
        "extracted": text[:500],
    })


# ---------------------------------------------------------------------------
# Agent Service (The Multi-Step Brain)
# ---------------------------------------------------------------------------
class AgentService:
    """
    Autonomous AI Agent that accepts high-level natural language tasks,
    chains multiple specialized tools iteratively (search -> inspect details -> compare -> synthesize),
    performs deterministic Python processing, and generates clear recommendations.
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.llm_model = llm_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.temperature = temperature
        self._llm: Optional[ChatGroq] = None

        # Registered tool capabilities
        self.tools = [
            search_amazon,
            get_product_details,
            compare_products,
            search_web,
            scrape_url,
            search_rag,
            extract_information,
        ]
        self.tools_by_name = {t.name: t for t in self.tools}

    def get_llm(self) -> ChatGroq:
        """Lazily initialize ChatGroq bound with tools."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please add GROQ_API_KEY to your .env file."
            )
        if self._llm is None:
            self._llm = ChatGroq(
                model=self.llm_model,
                temperature=self.temperature,
                api_key=groq_api_key,
                max_retries=3,
            )
        return self._llm

    async def run_task(self, task: str, max_steps: int = 6) -> Dict[str, Any]:
        """
        Executes a task through an autonomous multi-step iterative loop:
        1. Step 1 (Search): Calls search_amazon with relevant constraints.
        2. Step 2 (Deep Details): Calls get_product_details on top candidates to inspect voltage, warranty, timings.
        3. Step 3 (Comparison): Calls compare_products to obtain a deterministic breakdown & score verdict.
        4. Step 4 (Final Synthesis): Formulates a comprehensive markdown comparison table & advice.
        """
        llm = self.get_llm()
        llm_with_tools = llm.bind_tools(self.tools)

        system_instruction = (
            "You are an intelligent, autonomous multi-step research and shopping agent specialized in computer hardware.\n"
            "You have access to the following tools:\n"
            "- search_amazon: Search Amazon India (amazon.in) for products by query with budget (max_price), rating, "
            "capacity (capacity_gb), speed (speed_mhz), CL latency (cl_rating), and form factor (UDIMM for Desktop / SODIMM for Laptop) filters. "
            "Returns structured candidate products with deterministic Value Scores.\n"
            "- get_product_details: Fetches granular technical specifications for an Amazon India product URL (operating voltage, warranty, exact model, timings, bullets).\n"
            "- compare_products: Deterministically compares 2 to 4 product URLs side-by-side (specs, price/GB, latency, voltage, pros/cons, and category verdicts).\n"
            "- search_web: Alias for search_amazon.\n"
            "- scrape_url: Scrapes general webpage content.\n"
            "- search_rag: Search existing ChromaDB vector index for an indexed URL.\n"
            "- extract_information: Extract specific schema attributes from text.\n\n"
            "Autonomous Multi-Step Execution Strategy:\n"
            "When asked to find deals or recommend computer hardware (e.g., 'Find me the best DDR5 RAM deal'):\n"
            "1. Step 1: Call `search_amazon` ONCE with appropriate query and any specified constraints (e.g. form_factor='UDIMM' or 'SODIMM', capacity_gb, max_price). Do NOT repeat search_amazon calls.\n"
            "2. Step 2: From the returned candidate results, pick the top 2 candidates and call `get_product_details` for their URLs to inspect their operating voltage, warranty, and technical specifications.\n"
            "3. Step 3: Call `compare_products` with the candidate product URLs to generate a deterministic side-by-side comparison matrix and score verdict.\n"
            "4. Step 4: Synthesize your definitive recommendation with a Markdown comparison table, hardware justification, and purchase advice.\n\n"
            "Evaluation Rules:\n"
            "- Value Score vs Cheapest: The cheapest item (e.g. 8GB stick) is NOT necessarily the best deal. Use the deterministic 'value_score' to crown the best deal.\n"
            "- Form Factor Compatibility: Always distinguish Desktop (UDIMM) and Laptop (SODIMM) memory.\n"
            "- Hardware Justification: Highlight speed, CAS latency, voltage, and warranty in your explanation."
        )

        messages: List[Any] = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=task),
        ]

        step = 0
        steps_taken: List[Dict[str, Any]] = []
        tools_called_list: List[str] = []
        structured_data: Dict[str, Any] = {}
        all_products: List[Dict[str, Any]] = []
        detailed_products: List[Dict[str, Any]] = []
        comparison_data: Optional[Dict[str, Any]] = None
        final_answer: str = ""

        while step < max_steps:
            # Call LLM with automatic retry on TPM rate limit
            response = None
            for retry_i in range(3):
                try:
                    response = await llm_with_tools.ainvoke(messages)
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    if ("413" in err_msg or "rate_limit" in err_msg or "tpm" in err_msg or "429" in err_msg) and retry_i < 2:
                        await asyncio.sleep(5.0 * (retry_i + 1))
                    else:
                        raise

            messages.append(response)

            # Check if LLM finished and produced final content with no further tool calls
            if not response.tool_calls:
                final_answer = response.content if isinstance(response.content, str) else str(response.content)
                break

            # Execute tool calls emitted in this step
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]
                tools_called_list.append(tool_name)

                selected_tool = self.tools_by_name.get(tool_name)
                if selected_tool:
                    tool_output_str = await selected_tool.ainvoke(tool_args)
                    try:
                        parsed_out = json.loads(tool_output_str)
                    except Exception:
                        parsed_out = {"raw_output": tool_output_str}

                    if tool_name in ["search_amazon", "search_web"]:
                        structured_data = parsed_out
                        prods = parsed_out.get("products", [])
                        if prods:
                            all_products = prods
                    elif tool_name == "get_product_details":
                        detailed_products.append(parsed_out)
                    elif tool_name == "compare_products":
                        comparison_data = parsed_out

                    steps_taken.append({
                        "step": step + 1,
                        "tool": tool_name,
                        "arguments": tool_args,
                    })

                    messages.append(
                        ToolMessage(
                            tool_call_id=tool_id,
                            name=tool_name,
                            content=tool_output_str,
                        )
                    )
                else:
                    messages.append(
                        ToolMessage(
                            tool_call_id=tool_id,
                            name=tool_name,
                            content=json.dumps({"error": f"Tool '{tool_name}' is not recognized."}),
                        )
                    )

            step += 1

        # Fallback synthesis if final_answer is empty
        if not final_answer or not final_answer.strip():
            synthesis_messages = [
                SystemMessage(
                    content=(
                        "You are an expert computer hardware shopping advisor. "
                        "Based on the collected multi-step research data, provide a definitive, structured recommendation. "
                        "Include a Markdown comparison table, explain why the best value product wins using its Value Score, "
                        "and verify RAM form factor (UDIMM vs SODIMM), speed, CL latency, voltage, and warranty."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User Task: {task}\n\n"
                        f"Search Products:\n{json.dumps(all_products[:6], indent=2)}\n\n"
                        f"Detailed Products Inspected:\n{json.dumps(detailed_products, indent=2)}\n\n"
                        f"Comparison Results:\n{json.dumps(comparison_data, indent=2) if comparison_data else 'N/A'}\n\n"
                        "Please write the final recommendation."
                    )
                ),
            ]
            for retry_i in range(3):
                try:
                    synth_res = await llm.ainvoke(synthesis_messages)
                    final_answer = synth_res.content if isinstance(synth_res.content, str) else str(synth_res.content)
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    if ("413" in err_msg or "rate_limit" in err_msg or "tpm" in err_msg or "429" in err_msg) and retry_i < 2:
                        await asyncio.sleep(5.0 * (retry_i + 1))
                    else:
                        raise


        clean_final_answer = final_answer.strip()
        first_tool_called = tools_called_list[0] if tools_called_list else None
        first_tool_args = steps_taken[0]["arguments"] if steps_taken else None

        return {
            "task": task,
            "tool_called": first_tool_called,
            "tool_arguments": first_tool_args,
            "tools_called": tools_called_list,
            "steps_taken": steps_taken,
            "stats": structured_data.get("stats"),
            "products": all_products,
            "detailed_products": detailed_products,
            "comparison": comparison_data,
            "final_answer": clean_final_answer,
            "explanation": clean_final_answer,
            "model": self.llm_model,
        }


# Default singleton instance
default_agent_service = AgentService()


def get_agent_service() -> AgentService:
    return default_agent_service

