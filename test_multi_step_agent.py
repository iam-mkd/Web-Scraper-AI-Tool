import sys
import json
import asyncio
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from agent_service import get_agent_service, get_product_details, compare_products, search_amazon
from fastapi.testclient import TestClient
from main import app


async def test_tools_directly():
    print("=== 1. Testing get_product_details tool directly ===")
    sample_url = "https://www.amazon.in/CORSAIR-Vengeance-5200MHz-CL40-40-40-77-Desktop/dp/B0G61JZHBX"
    detail_res_str = await get_product_details.ainvoke({"url": sample_url})
    detail_res = json.loads(detail_res_str)
    
    print(f"Product Name: {detail_res.get('name')}")
    print(f"Price: ₹{detail_res.get('price')}")
    print(f"Voltage: {detail_res.get('voltage')}")
    print(f"Form Factor: {detail_res.get('form_factor')}")
    print(f"Tech Specs Keys: {list(detail_res.get('tech_specs', {}).keys())}")
    
    assert detail_res.get("price") is not None, "Expected valid price from detail scraper"
    assert "CORSAIR" in detail_res.get("name", "").upper(), "Expected Corsair in product name"
    print(">>> get_product_details tool test PASSED! <<<\n")

    print("=== 2. Testing compare_products tool directly ===")
    comp_res_str = await compare_products.ainvoke({"product_urls": [sample_url]})
    comp_res = json.loads(comp_res_str)
    
    print(f"Products Compared: {comp_res.get('products_compared')}")
    print(f"Verdict: {json.dumps(comp_res.get('verdict'), indent=2)}")
    
    assert comp_res.get("status") == "success", "Expected success status in comparison"
    assert "best_value" in comp_res.get("verdict", {}), "Expected best_value verdict"
    print(">>> compare_products tool test PASSED! <<<\n")


async def test_multi_step_agent():
    print("=== 3. Testing Autonomous Multi-Step Agent Execution ===")
    agent = get_agent_service()
    task = "Find me the best DDR5 RAM deal on Amazon"
    print(f"Executing Task: '{task}'")
    
    result = await agent.run_task(task, max_steps=5)
    
    print("\n--- Multi-Step Agent Execution Summary ---")
    print(f"Total Steps Taken: {len(result.get('steps_taken', []))}")
    for step_info in result.get("steps_taken", []):
        print(f"  Step {step_info.get('step')}: Tool -> {step_info.get('tool')}, Args -> {step_info.get('arguments')}")
        
    print(f"\nTools Called: {result.get('tools_called')}")
    print(f"Initial Candidates Scraped: {len(result.get('products', []))}")
    print(f"Detailed Products Inspected: {len(result.get('detailed_products', []))}")
    print(f"Comparison Available: {result.get('comparison') is not None}")
    
    print("\n--- Final Synthesized Recommendation ---")
    print(result.get("final_answer")[:500] + ("..." if len(result.get("final_answer", "")) > 500 else ""))
    
    assert len(result.get("steps_taken", [])) >= 1, "Expected at least 1 step taken"
    assert "search_amazon" in (result.get("tools_called") or []), "Expected search_amazon in tools_called"
    assert result.get("final_answer"), "Expected non-empty final answer"
    print("\n>>> Autonomous Multi-Step Agent Test PASSED! <<<\n")


def test_api_endpoints():
    print("=== 4. Testing FastAPI Multi-Step Endpoints ===")
    client = TestClient(app)
    
    # Test /api/research endpoint
    payload = {"task": "Find me the best DDR5 RAM deal on Amazon"}
    resp = client.post("/api/research", json=payload)
    print(f"/api/research Status: {resp.status_code}")
    assert resp.status_code == 200, f"Error: {resp.text}"
    
    data = resp.json()
    print(f"Returned Tools Called: {data.get('tools_called')}")
    print(f"Returned Steps Taken: {len(data.get('steps_taken', []))}")
    print(f"Returned Products Count: {len(data.get('products', []))}")
    print(f"Final Answer Preview: {data.get('final_answer')[:250]}...")
    
    assert len(data.get("products", [])) > 0, "Expected products in research response"
    assert data.get("final_answer"), "Expected final answer in research response"
    print("\n>>> FastAPI Endpoints Test PASSED! <<<\n")


if __name__ == "__main__":
    import time
    asyncio.run(test_tools_directly())
    print("Pacing for Groq TPM rate limits (sleeping 6s)...")
    time.sleep(6.0)
    asyncio.run(test_multi_step_agent())
    print("Pacing for Groq TPM rate limits (sleeping 10s)...")
    time.sleep(10.0)
    test_api_endpoints()


