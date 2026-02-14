"""Tests for the FastAPI API server."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agent.api.server import create_api
from agent.core.events import HumanInputEvent


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

    def test_submit_input(self, client, event_queue):
        """Test POST /api/bot queues a HumanInputEvent."""
        response = client.post(
            "/api/bot",
            json={
                "message": "Hello from test",
                "session_id": "test-session",
                "message_id": "test-message-id",
            },
        )
        assert response.status_code == 200
        assert response.json() == {"status": "queued"}

        # Verify event was queued
        assert not event_queue.empty()
        event = event_queue.get_nowait()
        assert isinstance(event, HumanInputEvent)
        assert event.message == "Hello from test"

    def test_submit_input_validation_error(self, client):
        """Test POST /api/bot with invalid body returns 422."""
        response = client.post("/api/bot", json={})
        assert response.status_code == 422

    def test_submit_input_empty_message(self, client, event_queue):
        """Test POST /api/bot with empty message still queues."""
        response = client.post(
            "/api/bot",
            json={
                "message": "",
                "session_id": "empty-message-session",
                "message_id": "empty-message-id",
            },
        )
        assert response.status_code == 200
        assert not event_queue.empty()
