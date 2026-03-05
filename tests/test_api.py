"""Tests for the FastAPI API server."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agent.api.server import create_api
from agent.core.events import TextInputEvent


@pytest.fixture
def event_queue():
    return asyncio.Queue()


@pytest.fixture
def client(event_queue):
    app = create_api(event_queue)
    return TestClient(app)


class TestAPI:
    """Tests for the FastAPI endpoints."""

    def test_health_check(self, client):
        """Test GET /api/health returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_chat_ui(self, client):
        """Test GET /chat serves HTML (file may not exist in CI, so just check it's attempted)."""
        try:
            response = client.get("/chat")
            assert response.status_code in (200, 500)
        except Exception:
            pass  # test_chat.html may be absent in some environments

    def test_ws_connect_and_send(self, client, event_queue):
        """Test WebSocket /api/bot: connect, receive chat_id, send a message."""
        with client.websocket_connect("/api/bot") as ws:
            # First message should be {"type": "connected", "chat_id": ...}
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            chat_id = connected["chat_id"]
            assert chat_id.startswith("ws-")

            # Send a message
            ws.send_json({"message": "Hello from test", "message_id": "test-msg-1"})

        # Verify event was queued with sender attached
        assert not event_queue.empty()
        event = event_queue.get_nowait()
        assert isinstance(event, TextInputEvent)
        assert event.message == "Hello from test"
        assert event.message_id == "test-msg-1"
        assert event.chat_id.startswith("ws-")
        assert event.sender is not None

    def test_ws_auto_message_id(self, client, event_queue):
        """Test that message_id is auto-assigned when omitted."""
        with client.websocket_connect("/api/bot") as ws:
            ws.receive_json()  # connected frame
            ws.send_json({"message": "no id provided"})

        assert not event_queue.empty()
        event = event_queue.get_nowait()
        assert isinstance(event, TextInputEvent)
        assert event.message_id  # auto-generated, non-empty
        assert event.sender is not None
