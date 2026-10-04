import asyncio
import json
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from agent_service import get_agent_service
from fastapi.testclient import TestClient
from main import app


async def test_agent_direct():
    print("=== Testing AgentService directly ===")
    agent = get_agent_service()
    task = "Find me DDR5 RAM deals on Amazon under 50000"
    print(f"Task: {task}")
    
    result = await agent.run_task(task)
    print("\n--- Agent Execution Output ---")
    print(f"Tool Called: {result.get('tool_called')}")
    print(f"Tool Arguments: {json.dumps(result.get('tool_arguments'), indent=2)}")
    print(f"Deterministic Stats: {json.dumps(result.get('stats'), indent=2)}")
    print(f"Matching Products Found: {len(result.get('products', []))}")
    if result.get("products"):
        print(f"Sample Product: {json.dumps(result['products'][0], indent=2)}")
    print("\n--- LLM Explanation ---")
    print(result.get("explanation"))
    
    assert result.get("tool_called") in ("search_web", "search_amazon"), f"Unexpected tool: {result.get('tool_called')}"
    assert len(result.get("products", [])) > 0, "Expected at least 1 matching product"
    assert "cheapest_product" in (result.get("stats") or {}), "Expected cheapest_product in stats"
    print("\n>>> Direct Agent Service Test Passed! <<<\n")


def test_agent_api():
    print("=== Testing /api/agent/task API endpoint ===")
    client = TestClient(app)
    payload = {"task": "Find me DDR5 RAM deals on Amazon under 50000"}
    
    response = client.post("/api/agent/task", json=payload)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200, f"API error: {response.text}"
    
    data = response.json()
    print(f"Response Task: {data.get('task')}")
    print(f"Tool Called: {data.get('tool_called')}")
    print(f"Products Count: {len(data.get('products', []))}")
    print(f"Explanation (preview): {data.get('explanation')[:300]}...")
    
    assert data.get("tool_called") in ("search_web", "search_amazon")
    assert len(data.get("products", [])) > 0
    assert data.get("explanation")
    print("\n>>> FastAPI Agent Endpoint Test Passed! <<<\n")


if __name__ == "__main__":
    import time
    asyncio.run(test_agent_direct())
    time.sleep(2.0)
    test_agent_api()
