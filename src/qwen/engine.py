import functools
import logging
import torch
import asyncio
from typing import AsyncIterator

from qwen.model import QwenForCausalLM
from qwen.cache import init_kv_cache2
from qwen.sampling import apply_penalties, sample

logger = logging.getLogger(__name__)


@torch.no_grad()
def _decode_step(model: QwenForCausalLM,
                    latest_tokens: torch.Tensor,
                    prompt_tokens: torch.Tensor,
                    output_tokens: torch.Tensor,
                    temperature: torch.Tensor | None = None,
                    top_k: torch.Tensor | None = None,
                    top_p: torch.Tensor | None = None,
                    rep_pen: torch.Tensor | None = None,
                    freq_pen: torch.Tensor | None = None,
                    pres_pen: torch.Tensor | None = None,
                    cache: list | None = None):
    output = model.forward(latest_tokens, cache)
    cache = output.past_key_values

    # handle default parameters
    bsz, _ = latest_tokens.size()
    if temperature is None:
        temperature = torch.full(
            size=(bsz,), 
            fill_value=model.config.temperature, 
            device=model.device,
            dtype=model.dtype
        )
    if top_k is None:
        top_k = torch.full(
            size=(bsz,), 
            fill_value=model.config.top_k, 
            device=model.device,
            dtype=torch.int64
        )
    if top_p is None:
        top_p = torch.full(
            size=(bsz,), 
            fill_value=model.config.top_p, 
            device=model.device,
            dtype=model.dtype
        )
    if rep_pen is None:
        rep_pen = torch.full(
            size=(bsz,), 
            fill_value=model.config.repetition_penalty, 
            device=model.device,
            dtype=model.dtype
        )
    if freq_pen is None:
        freq_pen = torch.full(
            size=(bsz,), 
            fill_value=model.config.frequency_penalty, 
            device=model.device,
            dtype=model.dtype
        )
    if pres_pen is None:
        pres_pen = torch.full(
            size=(bsz,), 
            fill_value=model.config.presence_penalty, 
            device=model.device,
            dtype=model.dtype
        )

    output.logits = apply_penalties(output.logits, prompt_tokens, output_tokens, rep_pen, freq_pen, pres_pen, model.config.vocab_size)
    if model.config.do_sample:
        next_tokens = sample(output.logits, temperature=temperature, top_k=top_k, top_p=top_p)
    else:
        next_tokens = output.logits.argmax(dim=-1)
    
    return next_tokens.unsqueeze(0), cache

def generate(model: QwenForCausalLM,
                input_ids: torch.Tensor,
                temperature: torch.Tensor | None = None,
                top_k: torch.Tensor | None = None,
                top_p: torch.Tensor | None = None,
                rep_pen: torch.Tensor | None = None,
                freq_pen: torch.Tensor | None = None,
                pres_pen: torch.Tensor | None = None,
                max_new_tokens: int = 300) -> torch.Tensor:
    bsz, seq_len = input_ids.size()
    if bsz > 1:
        raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
    
    cache = init_kv_cache2(model.config, bsz, model.device, model.dtype)
    
    latest_token_id = input_ids[0, -1]  # todo: only for bsz=1
    latest_tokens = input_ids
    output_tokens = input_ids[:, :0]    # empty tensor, [bsz, 0]
    max_new = min(model.config.max_position_embeddings - seq_len, max_new_tokens)
    while latest_token_id not in model.config.eos_token_id and output_tokens.size(-1) < max_new:
        latest_tokens, cache = _decode_step(model=model,
                                            latest_tokens=latest_tokens,
                                            prompt_tokens=input_ids,
                                            output_tokens=output_tokens,
                                            temperature=temperature,
                                            top_k=top_k,
                                            top_p=top_p,
                                            rep_pen=rep_pen,
                                            freq_pen=freq_pen,
                                            pres_pen=pres_pen,
                                            cache=list(cache) if cache is not None else None)
        output_tokens = torch.cat([output_tokens, latest_tokens], -1)
        latest_token_id = latest_tokens[0, 0].item()

    return torch.cat([input_ids, output_tokens], -1)

async def async_generate(model: QwenForCausalLM,
                        input_ids: torch.Tensor,
                        temperature: torch.Tensor | None = None,
                        top_k: torch.Tensor | None = None,
                        top_p: torch.Tensor | None = None,
                        rep_pen: torch.Tensor | None = None,
                        freq_pen: torch.Tensor | None = None,
                        pres_pen: torch.Tensor | None = None,
                        max_new_tokens: int = 300) -> AsyncIterator[torch.Tensor]:
    bsz, seq_len = input_ids.size()
    if bsz > 1:
        raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
    
    cache = init_kv_cache2(model.config, bsz, model.device, model.dtype)
    output_tokens = input_ids[:, :0]    # empty tensor, [bsz, 0]
    latest_tokens = input_ids   # shape [bsz, seq_len]
    yield latest_tokens

    latest_token_id = input_ids[0, -1]  # todo: only for bsz=1
    loop = asyncio.get_running_loop()
    max_new = min(model.config.max_position_embeddings - seq_len, max_new_tokens)
    while latest_token_id not in model.config.eos_token_id and output_tokens.size(-1) < max_new:
        latest_tokens, cache = await loop.run_in_executor(None, 
                                                          functools.partial(
                                                                _decode_step,
                                                                model=model,
                                                                latest_tokens=latest_tokens,
                                                                prompt_tokens=input_ids,
                                                                output_tokens=output_tokens,
                                                                temperature=temperature,
                                                                top_k=top_k,
                                                                top_p=top_p,
                                                                rep_pen=rep_pen,
                                                                freq_pen=freq_pen,
                                                                pres_pen=pres_pen,
                                                                cache=list(cache) if cache is not None else None)
        )
        output_tokens = torch.cat([output_tokens, latest_tokens], -1)
        latest_token_id = latest_tokens[0, 0].item()

        yield latest_tokens
