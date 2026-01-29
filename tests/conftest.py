"""Pytest fixtures and configuration."""

import pytest
import httpx
from httpx import ASGITransport

from src.ai_assistant.main import get_app


@pytest.fixture
async def client() -> httpx.AsyncClient:
    """FastAPI async test client."""
    app = get_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
