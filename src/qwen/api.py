from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

from qwen.model import QwenModel
from qwen.config import ModelConfig
from qwen.engine import async_generate


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

# constant variables
MODEL_DIR = "../qwen2.5-0.5b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    model = QwenModel(cfg)
    app.state.model_client = model
    yield

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float | None = None
    top_k: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

app = FastAPI(lifespan=lifespan)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

logger.info("Qwen HTTP service is starting...")

def get_model_client(request: Request) -> QwenModel:
    return request.app.state.model_client

@app.get("/health")
async def stream():
    async def gen():
        for w in ["OK\n"]:
            yield w
    return StreamingResponse(gen(), media_type="text/plain")


def _generate_stream_imp(req: GenRequest, model: QwenModel,
                         prefix: str="data: ", suffix: str="\n\n"):
    input_ids = tokenizer(req.prompt, return_tensors="pt").input_ids.to(model.device)

    async def sse():
        async for token in async_generate(model, input_ids, max_new_tokens=req.max_new_tokens):
            piece = tokenizer.decode(token[0])
            yield prefix+piece+suffix
        yield prefix+"[DONE]"+suffix

    return StreamingResponse(sse(), media_type="text/plain")

@app.post("/generate_stream")
async def generate_stream(req: GenRequest, model: QwenModel = Depends(get_model_client)):
    return _generate_stream_imp(req, model)

# curl -N -X POST
@app.post("/generate_stream_plain")
async def generate_stream_plain(req: GenRequest, model: QwenModel = Depends(get_model_client)):
    return _generate_stream_imp(req, model, prefix="", suffix="")

