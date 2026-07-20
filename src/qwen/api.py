from sys import prefix

from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from pydantic import BaseModel, computed_field, model_validator
from contextlib import asynccontextmanager
import logging
import orjson
from collections.abc import Callable

from qwen.model import QwenForCausalLM
from qwen.config import ModelConfig
from qwen.engine import async_generate
from qwen.constants import MODEL_DIR


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
    app.state.model_client = model
    yield

class GenRequest(BaseModel):
    prompt: str | None = None
    prompts: list[str] | None = None
    max_new_tokens: int = 100
    temperature: float | None = None

    @model_validator(mode="after")
    def _check_prompt_source(self) -> "GenRequest":
        if (self.prompt is None) == (self.prompts is None):
            raise ValueError("exactly one of 'prompt' or 'prompts' must be set")
        if self.prompts is not None and not self.prompts:
            # empty list would silently produce a zero-request batch
            raise ValueError("'prompts' must not be empty")
        return self

    @computed_field
    @property
    def batch(self) -> list[str]:
        return [self.prompt] if self.prompt is not None else self.prompts

app = FastAPI(lifespan=lifespan)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

logger.info("Qwen HTTP service is starting...")

def get_model_client(request: Request) -> QwenForCausalLM:
    return request.app.state.model_client

@app.get("/health")
async def stream():
    async def gen():
        for w in ["OK\n"]:
            yield w
    return StreamingResponse(gen(), media_type="text/plain")


def _generate_stream_imp2(req: GenRequest, model: QwenForCausalLM,
                         prefix: str="data: ", suffix: str="\n\n"):
    input_ids = tokenizer(req.batch).input_ids    # raw tokenizer output, a list of variable-length id sequences

    async def sse():
        try:
            async for token in async_generate(model, input_ids, max_new_tokens=req.max_new_tokens):
                piece = tokenizer.decode(token[0])
                yield prefix+piece+suffix
            yield prefix+"[DONE]"+suffix
        finally:
            # todo: release KV cache slots for client-disconnect scenario.
            pass

    return StreamingResponse(sse(), media_type="text/event-stream")


def _generate_stream_imp(req: GenRequest, model: QwenForCausalLM,
                         payload_generate: Callable[[int, str], bytes] = lambda i, text: orjson.dumps({"index": i, "text": text}),
                         prefix: str=b"data: ", suffix: str=b"\n\n"):
    input_ids = tokenizer(req.batch).input_ids    # raw tokenizer output, a list of variable-length id sequences

    async def sse():
        try:
            async for step_tokens in async_generate(model, input_ids, max_new_tokens=req.max_new_tokens):
                for i, token in enumerate(step_tokens):
                    text = tokenizer.decode(token)
                    payload = payload_generate(i, text)
                    yield prefix + payload + suffix
            yield prefix+b"[DONE]\n\n"        # normal completion only
        finally:
            # todo: release KV cache slots for client-disconnect scenario.
            pass

    return StreamingResponse(sse(), media_type="text/event-stream")

@app.post("/generate_stream")
async def generate_stream(req: GenRequest, model: QwenForCausalLM = Depends(get_model_client)):
    return _generate_stream_imp(
        req, model,
        payload_generate=lambda i, text: orjson.dumps({"index": i, "text": text}),
        prefix=b"data: ", suffix=b"\n\n",
    )

# curl -N -X POST
@app.post("/generate_stream_plain")
async def generate_stream_plain(req: GenRequest, model: QwenForCausalLM = Depends(get_model_client)):
    assert req.prompt is not None, "Prompt is required"

    return _generate_stream_imp(
        req, model,
        payload_generate=lambda i, text: text.encode(),
        prefix=b"", suffix=b"",
    )


