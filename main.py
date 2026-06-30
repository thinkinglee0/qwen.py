from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from pydantic import BaseModel
from contextlib import asynccontextmanager

from qwen import QwenModel
from config import ModelConfig

# constant variables
MODEL_DIR = "../qwen2.5-0.5b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ModelConfig.from_pretrained(MODEL_DIR)
    model = QwenModel(cfg)
    app.state.model_client = model                   # 存到 app.state,startup 时 load 一次
    yield

class GenRequest(BaseModel):
    prompt: str
    max_len: int = 100
    # temperature: float = 0.7

app = FastAPI(lifespan=lifespan)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

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
        async for token in model.generate_stream(input_ids, req.max_len):
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

