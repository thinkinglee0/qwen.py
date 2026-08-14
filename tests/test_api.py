# tests/test_engine.py

import logging
import pytest_asyncio
import pytest
import orjson

from httpx import ASGITransport, AsyncClient
from qwen.api import app, get_driver
from constants import PROMPT_CLASSICAL

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(loop_scope="module")
async def api_client(target_driver):
    app.dependency_overrides[get_driver] = lambda: target_driver
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
    async with api_client.stream("POST", "/generate_stream", json={"prompt": PROMPT_CLASSICAL}) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                json_str = line.removeprefix("data: ").strip()
                if json_str == "[DONE]":
                    chunks.append(json_str)
                    break

                raw = orjson.loads(json_str)
                assert "text" in raw
                chunks.append(raw["text"])
    
    logger.info(f"fastapi output: |{''.join(chunks)}|")
    assert chunks and chunks[-1] == "[DONE]"