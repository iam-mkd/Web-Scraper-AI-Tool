import asyncio
import json
from fastapi.testclient import TestClient
from main import app
from agent_service import get_agent_service, analyze_task_intent

client = TestClient(app)


def test_intent_analyzer():
    print("=== 0. Testing Query Intent & Considerations Analyzer ===")
    
    # 1. Raw performance query
    res1 = analyze_task_intent("Find me the absolute best DDR5 RAM kit, I don't care about the price")
    assert res1["optimization"] == "Raw performance"
    assert res1["budget"] == "None"
    assert res1["form_factor"] == "UDIMM"
    assert "raw performance" in res1["intent_summary"]
    print("  [PASS] Raw performance intent parsed: Optimization=Raw performance, Budget=None")

    # 2. Budget limited query
    res2 = analyze_task_intent("Find me the best DDR5 RAM deals on Amazon under ₹50,000")
    assert res2["optimization"] == "Best value (Perf / ₹)"
    assert res2["budget"] == "₹50,000"
    assert res2["form_factor"] == "UDIMM"
    print("  [PASS] Budget intent parsed: Optimization=Best value, Budget=Rs. 50,000")

    # 3. Laptop SODIMM query
    res3 = analyze_task_intent("Find me 32GB DDR5 SODIMM laptop RAM")
    assert res3["form_factor"] == "SODIMM"
    assert res3["capacity"] == "32GB"
    print("  [PASS] Form factor intent parsed: Form factor=SODIMM, Capacity=32GB")


def test_static_routes():
    print("\n=== 1. Testing Web UI Static Routes ===")
    
    # 1. Root HTML
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert "text/html" in resp_root.headers.get("content-type", "")
    assert "Autonomous Web Scraper AI" in resp_root.text
    assert "Finds candidate products from Amazon" in resp_root.text
    assert "id=\"traceTimeline\"" in resp_root.text
    print("  [PASS] GET / serves index.html with clean Amazon description & trace UI")

    # 2. Stylesheet
    resp_css = client.get("/static/styles.css")
    assert resp_css.status_code == 200
    assert "text/css" in resp_css.headers.get("content-type", "")
    assert ".trace-considerations" in resp_css.text
    assert ".trace-params-block" in resp_css.text
    print("  [PASS] GET /static/styles.css serves considerations & tool params styles")

    # 3. Frontend App JavaScript
    resp_js = client.get("/static/app.js")
    assert resp_js.status_code == 200
    assert "javascript" in resp_js.headers.get("content-type", "")
    assert "handleStreamEvent" in resp_js.text
    print("  [PASS] GET /static/app.js serves streaming trace handler")

    # 4. JSON API Root verification
    resp_json = client.get("/api")
    assert resp_json.status_code == 200
    assert resp_json.json().get("name") == "Web Scraper AI Tool API"
    print("  [PASS] GET /api serves JSON API root metadata")


async def test_stream_generator_mock():
    print("\n=== 2. Testing Streaming Generator Mechanics ===")
    agent = get_agent_service()
    
    # Check that run_task_stream yields start and planning immediately
    stream_gen = agent.run_task_stream("Find me the absolute best DDR5 RAM kit, I don't care about the price")
    events = []
    
    async for chunk in stream_gen:
        events.append(chunk)
        if len(events) >= 2:
            break
            
    assert len(events) >= 2
    assert isinstance(events[0], dict)
    assert events[0].get("type") == "start"
    assert events[1].get("type") == "planning"
    assert events[1].get("considerations", {}).get("Optimization") == "Raw performance"
    assert events[1].get("considerations", {}).get("Budget") == "None"
    print(f"  [PASS] Generator emitted planning with considerations: {events[1]['considerations']}")


def test_sse_endpoint_stream():
    print("\n=== 3. Testing POST /api/agent/stream SSE Endpoint ===")
    with client.stream("POST", "/api/agent/stream", json={"task": "Find top DDR5 RAM"}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        
        event_types = []
        for line in response.iter_lines():
            line_str = line.strip() if isinstance(line, str) else line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                try:
                    payload = json.loads(line_str[6:])
                    etype = payload.get("type")
                    event_types.append(etype)
                    print(f"  -> SSE event emitted: type={etype}")
                    if len(event_types) >= 3:
                        break
                except Exception:
                    pass
                    
        assert len(event_types) >= 2
        assert "start" in event_types
        assert "planning" in event_types
        print(f"  [PASS] Stream successfully delivered start & planning events: {event_types}")


if __name__ == "__main__":
    print("Starting Web UI and Trace Polish Verification...")
    test_intent_analyzer();
    test_static_routes()
    asyncio.run(test_stream_generator_mock())
    test_sse_endpoint_stream()
    print("\n[SUCCESS] All Web UI & Trace Polish tests completed successfully!")
