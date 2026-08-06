import numpy as np
from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi import Request as HTTPRequest
from transformers import AutoTokenizer
from pydantic import BaseModel, computed_field, model_validator
from contextlib import asynccontextmanager
import logging
import orjson
from collections.abc import Callable

from qwen.model import QwenForCausalLM
from qwen.config import ModelConfig
from qwen.engine import async_generate, ServingDriver, LLMEngine
from qwen.constants import MODEL_DIR
from qwen.sampling import Sampling
from qwen.scheduler import StaticScheduler

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} [{levelname}] {filename}:{lineno} - {message}",
    style="{",
    handlers=[
        logging.FileHandler("log/app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    model = QwenForCausalLM(cfg)
    scheduler = StaticScheduler(cfg)
    engine = LLMEngine(model, scheduler)
    driver = ServingDriver(engine)
    driver.start()      # start the run_loop in a separate thread
    app.state.driver = driver
    yield

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    sampling: Sampling = Sampling()

app = FastAPI(lifespan=lifespan)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

logger.info("Qwen HTTP service is starting...")

def get_driver(request: Request) -> ServingDriver:
    return request.app.state.driver

@app.get("/health")
async def stream():
    async def gen():
        for w in ["OK\n"]:
            yield w
    return StreamingResponse(gen(), media_type="text/plain")


def _generate_stream_imp(http_req: HTTPRequest, req: GenRequest, driver: ServingDriver,
                         payload_generate: Callable[[str], bytes] = lambda text: orjson.dumps({"text": text}),
                         prefix: bytes=b"data: ", suffix: bytes=b"\n\n"):
    input_ids = tokenizer(req.prompt).input_ids     # convert prompt to token ids

    async def sse():
        try:
            async for token in async_generate(driver, input_ids, req.sampling):
                if await http_req.is_disconnected():      # client closed connection
                    break
                text = tokenizer.decode(token)
                payload = payload_generate(text)
                yield prefix + payload + suffix
            yield prefix+b"[DONE]\n\n"        # normal completion only
        finally:
            # todo: release KV cache slots for client-disconnect scenario.
            pass

    return StreamingResponse(sse(), media_type="text/event-stream")

@app.post("/generate_stream")
async def generate_stream(http_req: HTTPRequest, req: GenRequest, driver: ServingDriver = Depends(get_driver)):
    return _generate_stream_imp(
        http_req, req, driver,
        payload_generate=lambda text: orjson.dumps({"text": text}),
        prefix=b"data: ", suffix=b"\n\n",
    )

# curl -N -X POST
@app.post("/generate_stream_plain")
async def generate_stream_plain(http_req: HTTPRequest, req: GenRequest, driver: ServingDriver = Depends(get_driver)):
    assert req.prompt, "Prompt is required"

    return _generate_stream_imp(
        http_req, req, driver,
        payload_generate=lambda text: text.encode(),
        prefix=b"", suffix=b"",
    )


