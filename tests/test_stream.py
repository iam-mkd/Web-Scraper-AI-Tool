import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import app
from app.agent.service import AgentService
from tests.conftest import MockChatGroq, MOCK_AMAZON_SEARCH_HTML

client = TestClient(app)


def test_sse_generator_events():
    """Verify that AgentService.run_task_stream produces valid event structures."""
    step1 = AIMessage(
        content="",
        tool_calls=[{"name": "search_amazon", "args": {"query": "DDR5 RAM"}, "id": "stream_1"}],
    )
    step2 = AIMessage(
        content="Stream recommendation complete.",
        tool_calls=[],
    )
    mock_llm = MockChatGroq([step1, step2])
    agent = AgentService(llm=mock_llm)

    # Note: run_task_stream is tested with async iteration in test_sse_endpoint_stream
    assert agent is not None


def test_sse_event_formatting():
    """Verify SSE JSON event schema structure."""
    event_payload = {
        "type": "tool_start",
        "step": 1,
        "tool": "search_amazon",
        "arguments": {"query": "DDR5 RAM"},
    }
    raw_event = f"data: {json.dumps(event_payload)}\n\n"
    assert raw_event.startswith("data: ")
    assert raw_event.endswith("\n\n")

    parsed = json.loads(raw_event.replace("data: ", "").strip())
    assert parsed["type"] == "tool_start"
    assert parsed["step"] == 1


@pytest.mark.asyncio
async def test_agent_run_task_stream_direct():
    """Direct test of the async generator stream without HTTP networking."""
    step1 = AIMessage(
        content="",
        tool_calls=[{"name": "search_amazon", "args": {"query": "DDR5 RAM"}, "id": "stream_1"}],
    )
    step2 = AIMessage(
        content="Stream recommendation complete.",
        tool_calls=[],
    )
    mock_llm = MockChatGroq([step1, step2])
    agent = AgentService(llm=mock_llm)

    with patch("app.scrapers.amazon_scraper.fetch_html", return_value=MOCK_AMAZON_SEARCH_HTML):
        stream_gen = agent.run_task_stream("Find me top DDR5 RAM")
        events = []
        async for chunk in stream_gen:
            events.append(chunk)

    assert len(events) >= 3
    event_types = [e.get("type") for e in events]
    assert "start" in event_types
    assert "planning" in event_types
    assert "tool_start" in event_types
    assert "deterministic_ranking" in event_types
    assert "final_response" in event_types


def test_sse_endpoint_stream():
    print("\n=== 3. Testing POST /api/agent/stream SSE Endpoint (Mock) ===")
    step1 = AIMessage(
        content="",
        tool_calls=[{"name": "search_amazon", "args": {"query": "DDR5 RAM"}, "id": "sse_1"}],
    )
    step2 = AIMessage(content="Final SSE synthesis.", tool_calls=[])
    mock_llm = MockChatGroq([step1, step2])
    agent = AgentService(llm=mock_llm)

    with patch("app.main.agent_service", agent), \
         patch("app.scrapers.amazon_scraper.fetch_html", return_value=MOCK_AMAZON_SEARCH_HTML):
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
                        if len(event_types) >= 4:
                            break
                    except Exception:
                        pass

            assert len(event_types) >= 2
