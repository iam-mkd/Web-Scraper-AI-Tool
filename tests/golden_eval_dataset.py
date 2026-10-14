"""
Golden Evaluation Dataset for Computer Hardware Agent

A curated set of 30 representative evaluation cases across 5 categories:
- 8 Intent and constraint extraction cases
- 6 Tool selection and argument cases
- 6 Product ranking and comparison cases
- 6 Grounding and missing-information cases
- 4 Edge cases (impossible budgets, contradictory requests, out-of-domain)

Each case defines:
- task: Natural language query submitted by user
- expected: Ground truth constraints, permitted/forbidden tools, ranking criteria, grounding facts
- grading_notes: Explains what specific agent capability is under test
"""
from typing import Dict, Any, List, Optional

GOLDEN_EVAL_DATASET: List[Dict[str, Any]] = [
    # =========================================================================
    # Category 1: Intent & Constraint Extraction (8 cases)
    # Tests deterministic extraction of budget, optimization, form factor, capacity
    # =========================================================================
    {
        "id": "intent_01",
        "category": "intent_extraction",
        "task": "Find me the best DDR5 RAM deal on Amazon under 10000",
        "expected": {
            "optimization": "Best value (Perf / ₹)",
            "budget": "₹10,000",
            "budget_val": 10000.0,
            "form_factor": "UDIMM",
            "capacity": None,
            "permitted_tools": ["search_amazon"],
            "forbidden_tools": ["scrape_url"],
            "tool_args": {"query": "DDR5 RAM", "max_price": 10000.0},
        },
        "grading_notes": "Tests standard budget extraction (numeric 10000), default desktop UDIMM form factor, and best value optimization.",
    },
    {
        "id": "intent_02",
        "category": "intent_extraction",
        "task": "I need 32GB laptop DDR5 memory",
        "expected": {
            "optimization": "Best value (Perf / ₹)",
            "budget": "None",
            "budget_val": None,
            "form_factor": "SODIMM",
            "capacity": "32GB",
            "permitted_tools": ["search_amazon"],
            "forbidden_tools": [],
            "tool_args": {"query": "32GB DDR5 RAM", "capacity_gb": 32, "form_factor": "SODIMM"},
        },
        "grading_notes": "Tests extraction of SODIMM form factor from 'laptop' keyword, 32GB capacity, and unbound budget.",
    },
    {
        "id": "intent_03",
        "category": "intent_extraction",
        "task": "What's the cheapest DDR5 RAM available for my desktop?",
        "expected": {
            "optimization": "Lowest price",
            "budget": "None",
            "budget_val": None,
            "form_factor": "UDIMM",
            "capacity": None,
            "permitted_tools": ["search_amazon"],
            "forbidden_tools": [],
            "tool_args": {"optimization": "cheapest", "sort_by": "price_asc"},
        },
        "grading_notes": "Tests optimization detection for 'cheapest', mapping to Lowest price intent.",
    },
    {
        "id": "intent_04",
        "category": "intent_extraction",
        "task": "I don't care about the price, give me the fastest DDR5 kit for overclocking",
        "expected": {
            "optimization": "Raw performance",
            "budget": "None",
            "budget_val": None,
            "form_factor": "UDIMM",
            "capacity": None,
            "permitted_tools": ["search_amazon"],
            "forbidden_tools": [],
            "tool_args": {"optimization": "performance"},
        },
        "grading_notes": "Tests 'don't care about the price' rule which unbinds budget and maps optimization to Raw performance.",
    },
    {
        "id": "intent_05",
        "category": "intent_extraction",
        "task": "Top rated 16GB desktop RAM below 5000",
        "expected": {
            "optimization": "Highest customer rating",
            "budget": "₹5,000",
            "budget_val": 5000.0,
            "form_factor": "UDIMM",
            "capacity": "16GB",
            "permitted_tools": ["search_amazon"],
            "forbidden_tools": [],
            "tool_args": {"query": "16GB DDR5 RAM", "max_price": 5000.0, "capacity_gb": 16},
        },
        "grading_notes": "Tests 'top rated' mapping to Highest customer rating, 'below 5000' budget, and '16GB' capacity.",
    },
    {
        "id": "intent_06",
        "category": "intent_extraction",
        "task": "Find SODIMM DDR5 for my notebook below 8000",
        "expected": {
            "optimization": "Best value (Perf / ₹)",
            "budget": "₹8,000",
            "budget_val": 8000.0,
            "form_factor": "SODIMM",
            "capacity": None,
            "permitted_tools": ["search_amazon"],
            "forbidden_tools": [],
            "tool_args": {"form_factor": "SODIMM", "max_price": 8000.0},
        },
        "grading_notes": "Tests SODIMM detection from both 'SODIMM' and 'notebook' keywords with 'below 8000' budget.",
    },
    {
        "id": "intent_07",
        "category": "intent_extraction",
        "task": "Recommend the most reliable DDR5 RAM kit for a workstation PC",
        "expected": {
            "optimization": "Highest customer rating",
            "budget": "None",
            "budget_val": None,
            "form_factor": "UDIMM",
            "capacity": None,
            "permitted_tools": ["search_amazon"],
            "forbidden_tools": [],
            "tool_args": {"sort_by": "rating_desc"},
        },
        "grading_notes": "Tests 'most reliable' keyword triggering Highest customer rating and 'workstation PC' mapping to UDIMM.",
    },
    {
        "id": "intent_08",
        "category": "intent_extraction",
        "task": "Find fastest DDR5 RAM under 50k for my gaming rig",
        "expected": {
            "optimization": "Raw performance",
            "budget": "₹50,000",
            "budget_val": 50000.0,
            "form_factor": "UDIMM",
            "capacity": None,
            "permitted_tools": ["search_amazon"],
            "forbidden_tools": [],
            "tool_args": {"max_price": 50000.0, "form_factor": "UDIMM"},
        },
        "grading_notes": "Tests '50k' abbreviation parsing to 50000.0, 'highest speed' intent, and 'gaming rig' UDIMM detection.",
    },

    # =========================================================================
    # Category 2: Tool Selection & Arguments (6 cases)
    # Tests correct tool selection, chaining, and parameter hygiene
    # =========================================================================
    {
        "id": "tool_01",
        "category": "tool_selection",
        "task": "Search Amazon India for DDR5 6000MHz memory kits under 12000",
        "expected": {
            "permitted_tools": ["search_amazon", "get_product_details", "compare_products"],
            "forbidden_tools": ["scrape_url", "search_rag"],
            "tool_args": {"query": "DDR5 6000MHz", "max_price": 12000.0},
        },
        "grading_notes": "Agent must pick search_amazon rather than general web scraper or vector store, with accurate max_price.",
    },
    {
        "id": "tool_02",
        "category": "tool_selection",
        "task": "Inspect the operating voltage and timings of Corsair Vengeance DDR5 from Amazon: https://www.amazon.in/CORSAIR-Vengeance-5200MHz-CL40-40-40-77-Computer/dp/B0DM2PGP4N/ref=sr_1_3?sr=8-3",
        "permitted_tools": ["get_product_details", "search_amazon", "scrape_url", "search_rag"],
        "expected": {
            "permitted_tools": ["get_product_details", "search_amazon", "scrape_url", "search_rag"],
            "forbidden_tools": ["extract_information"],
            "tool_args": {"url": "https://www.amazon.in/CORSAIR-Vengeance-5200MHz-CL40-40-40-77-Computer/dp/B0DM2PGP4N/ref=sr_1_3?sr=8-3"},
        },
        "grading_notes": "Agent must invoke get_product_details or search_amazon followed by get_product_details to inspect granular specs.",
    },
    {
        "id": "tool_03",
        "category": "tool_selection",
        "task": "Compare Corsair Vengeance and G.Skill Trident Z5 side by side",
        "expected": {
            "permitted_tools": ["compare_products", "search_amazon", "search_amazon", "get_product_details", "get_product_details"],
            "forbidden_tools": ["scrape_url"],
            "tool_args": {"product_urls": ["https://www.amazon.in/CORSAIR-Vengeance-5200MHz-CL40-40-40-77-Computer/dp/B0DM2PGP4N/ref=sr_1_3?sr=8-3", "https://www.amazon.in/G-Skill-Trident-CL36-36-36-96-Desktop-Memory/dp/B09TPY273P/ref=sr_1_1?sr=8-1"]},
        },
        "grading_notes": "Agent must utilize compare_products to execute multi-product hardware matrix analysis.",
    },
    {
        "id": "tool_04",
        "category": "tool_selection",
        "task": "Scrape the hardware specification article at https://en.wikipedia.org/wiki/DDR5_SDRAM",
        "expected": {
            "permitted_tools": ["scrape_url","search_rag"],
            "forbidden_tools": ["search_amazon"],
            "tool_args": {"url": "https://en.wikipedia.org/wiki/DDR5_SDRAM"},
        },
        "grading_notes": "Direct URL request should correctly trigger scrape_url instead of shopping tools.",
    },
    {
        "id": "tool_05",
        "category": "tool_selection",
        "task": "Find 64GB DDR5 memory modules for server virtualization",
        "expected": {
            "permitted_tools": ["search_amazon", "search_web"],
            "forbidden_tools": [],
            "tool_args": {"query": "64GB DDR5 RAM", "capacity_gb": 64},
        },
        "grading_notes": "Ensures capacity_gb parameter is passed cleanly to search_amazon without schema violations.",
    },
    {
        "id": "tool_06",
        "category": "tool_selection",
        "task": "Search for SODIMM laptop RAM kits under 9000",
        "expected": {
            "permitted_tools": ["search_amazon","get_product_details", "compare_products"],
            "forbidden_tools": [],
            "tool_args": {"query": "SODIMM laptop RAM", "form_factor": "SODIMM", "max_price": 9000.0},
        },
        "grading_notes": "Verifies form_factor='SODIMM' argument is passed along with max_price to filter non-laptop memory.",
    },

    # =========================================================================
    # Category 3: Product Ranking & Comparison (6 cases)
    # Tests deterministic math, Value Score rankings, and budget enforcement
    # =========================================================================
    {
        "id": "rank_01",
        "category": "product_ranking",
        "task": "Find DDR5 RAM under 10000 and confirm all results respect the budget",
        "expected": {
            "budget_val": 10000.0,
            "budget_compliant": True,
            "min_products": 1,
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies strict budget ceiling enforcement: 100% of returned priced products must be <= 10000 INR.",
    },
    {
        "id": "rank_02",
        "category": "product_ranking",
        "task": "Find SODIMM DDR5 memory kits for a gaming laptop",
        "expected": {
            "form_factor": "SODIMM",
            "form_factor_compliant": True,
            "min_products": 1,
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies form factor filtering: 100% of returned items must have form_factor == 'SODIMM'.",
    },
    {
        "id": "rank_03",
        "category": "product_ranking",
        "task": "Identify the best value DDR5 RAM balancing speed, capacity, and price per GB",
        "expected": {
            "best_value_name_contains": "G.Skill Trident",
            "min_products": 2,
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies deterministic Value Score computation: 32GB 6000MHz @ 9499 wins over 16GB 4800MHz or 32GB 6400MHz @ 13999.",
    },
    {
        "id": "rank_04",
        "category": "product_ranking",
        "task": "Find the cheapest DDR5 memory module available",
        "expected": {
            "cheapest_name_contains": "Crucial 16GB",
            "min_products": 1,
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies budget pick selection: Crucial 16GB @ 3899 must be crowned as lowest initial purchase price.",
    },
    {
        "id": "rank_05",
        "category": "product_ranking",
        "task": "Find the highest performance DDR5 gaming kit regardless of price",
        "expected": {
            "best_value_name_contains": "G.Skill Trident",
            "min_products": 1,
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies performance ranking: 6400MHz CL32 takes top performance spot over 6000MHz CL36 and 4800MHz CL40.",
    },
    {
        "id": "rank_06",
        "category": "product_ranking",
        "task": "Perform a multi-item comparison and verify verdicts for best value, speed, and budget",
        "expected": {
            "min_products": 3,
            "permitted_tools": ["search_amazon", "compare_products"],
        },
        "grading_notes": "Verifies side-by-side comparison matrix output contains distinct verdicts for value, performance, and budget.",
    },

    # =========================================================================
    # Category 4: Grounding & Missing Information (6 cases)
    # Tests factual hallucination prevention, unpriced item handling, source citation
    # =========================================================================
    {
        "id": "ground_01",
        "category": "grounding",
        "task": "What DDR5 RAM kits did you find on Amazon India?",
        "expected": {
            "answer_must_contain": ["Corsair Vengeance", "Crucial"],
            "answer_must_not_contain": ["Samsung 980 Pro", "RTX 4090"],
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies that products cited in final answer exist in the real search results, not hallucinated models.",
    },
    {
        "id": "ground_02",
        "category": "grounding",
        "task": "How much does the Crucial 16GB DDR5 RAM cost on Amazon India?",
        "expected": {
            "answer_must_contain": [("3,899", "3899")],
            "answer_must_not_contain": ["5,000", "2,500"],
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies price grounding: must report exact scraped price (3,899 INR), not an invented round figure.",
    },
    {
        "id": "ground_03",
        "category": "grounding",
        "task": "Is the Crucial 32GB 5200MHz memory suitable for my desktop motherboard?",
        "expected": {
            "answer_must_contain": ["SODIMM", "laptop"],
            "answer_must_not_contain": ["is compatible with desktop UDIMM"],
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies form-factor grounding: Crucial 32GB 5200MHz is SODIMM (laptop) and incompatible with desktop UDIMM.",
    },
    {
        "id": "ground_04",
        "category": "grounding",
        "task": "What is the price and specs of the Kingston FURY Beast 32GB RAM?",
        "expected": {
            "answer_must_contain": [("unavailable", "out of stock", "no price")],
            "answer_must_not_contain": ["₹10,000", "₹8,000", "₹7,500"],
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies missing price handling: unpriced products must NOT have invented prices in the answer.",
    },
    {
        "id": "ground_05",
        "category": "grounding",
        "task": "Explain why the top recommended RAM was chosen using its Value Score",
        "expected": {
            "answer_must_contain": ["value score", "score", "/100"],
            "answer_must_not_contain": [],
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Verifies factual citation of deterministic Value Score metrics rather than subjective LLM praise.",
    },
    {
        "id": "ground_06",
        "category": "grounding",
        "task": "What is the operating voltage and warranty on the Corsair Vengeance DDR5 kit?",
        "expected": {
            "answer_must_contain": ["1.35", "lifetime"],
            "answer_must_not_contain": ["1.1V", "1.45V"],
            "permitted_tools": ["get_product_details", "search_amazon"],
        },
        "grading_notes": "Verifies technical specs grounding: extracts 1.35V voltage and Limited Lifetime Warranty from product details.",
    },

    # =========================================================================
    # Category 5: Edge Cases & Fault Tolerance (4 cases)
    # Tests graceful failure, empty results, out-of-domain requests, contradictions
    # =========================================================================
    {
        "id": "edge_01",
        "category": "edge_case",
        "task": "Find me 64GB DDR5 RAM for under 100 rupees",
        "expected": {
            "budget_val": 100.0,
            "answer_must_contain": ["no suitable products", "relaxing"],
            "answer_must_not_contain": ["Congratulations, found"],
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Impossible budget: search returns 0 products. Agent must explain gracefully without inventing products or crashing.",
    },
    {
        "id": "edge_02",
        "category": "edge_case",
        "task": "What is the weather in Mumbai today?",
        "expected": {
            "answer_must_contain": ["hardware", "shopping", "computer", ("unable", "not able")],
            "answer_must_not_contain": ["celsius", "rain forecast", "monsoon"],
            "permitted_tools": [],
        },
        "grading_notes": "Out-of-domain query: agent should clarify its computer hardware expertise rather than hallucinating live weather data.",
    },
    {
        "id": "edge_03",
        "category": "edge_case",
        "task": "   ",
        "expected": {
            "answer_must_contain": ["specify", ("assist", "help"), "ram", "product"],
            "answer_must_not_contain": [],
            "permitted_tools": [],
        },
        "grading_notes": "Whitespace / empty task: agent must handle gracefully without crashing or invoking null tools.",
    },
    {
        "id": "edge_04",
        "category": "edge_case",
        "task": "Find SODIMM DDR5 RAM for my standard desktop ATX motherboard",
        "expected": {
            "answer_must_contain": ["sodimm", ("udimm", "atx"), "incompatible", "laptop"],
            "answer_must_not_contain": ["perfectly compatible"],
            "permitted_tools": ["search_amazon"],
        },
        "grading_notes": "Contradictory user requirement: SODIMM is laptop form factor, desktop ATX motherboards use UDIMM. Must warn user.",
    },
]

# Quick Category Lookup Index
CATEGORY_INDEX: Dict[str, List[str]] = {}
for case in GOLDEN_EVAL_DATASET:
    CATEGORY_INDEX.setdefault(case["category"], []).append(case["id"])

CASE_MAP: Dict[str, Dict[str, Any]] = {case["id"]: case for case in GOLDEN_EVAL_DATASET}
