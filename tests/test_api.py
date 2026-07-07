# tests/test_engine.py

import logging
import pytest_asyncio
import pytest

from httpx import ASGITransport, AsyncClient
from qwen.api import app, get_model_client
from constants import CLASSIC_PROMPT

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(loop_scope="module")
async def api_client(target_model):
    app.dependency_overrides[get_model_client] = lambda: target_model
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_endpoint_health(api_client):
    r = await api_client.get("/health")
    assert r.status_code == 200

@pytest.mark.asyncio
async def test_endpoint_generate_stream(api_client):
    chunks = []
    async with api_client.stream("POST", "/generate_stream", json={"prompt": CLASSIC_PROMPT}) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line.removeprefix("data: ").strip())
    logger.info(f"fastapi output: |{chunks}|")
    assert chunks and chunks[-1] == "[DONE]"