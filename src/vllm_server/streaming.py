"""Streaming response generators for vLLM."""

import time
from typing import AsyncIterator
from vllm import SamplingParams
from .engine import get_engine


async def stream_completion(
    prompt: str, sampling_params: SamplingParams, request_id: str
) -> AsyncIterator[str]:
    """스트리밍 completion 생성"""
    engine = get_engine()
    previous_text = ""

    async for output in engine.generate(prompt, sampling_params, request_id):
        current_text = output.outputs[0].text
        delta = current_text[len(previous_text) :]
        previous_text = current_text

        if delta:
            chunk = {
                "id": request_id,
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "choices": [{"text": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {chunk}\n\n"

    # 종료 신호
    yield "data: [DONE]\n\n"


async def stream_chat_completion(
    prompt: str, sampling_params: SamplingParams, request_id: str
) -> AsyncIterator[str]:
    """스트리밍 chat completion 생성"""
    engine = get_engine()
    previous_text = ""

    async for output in engine.generate(prompt, sampling_params, request_id):
        current_text = output.outputs[0].text
        delta = current_text[len(previous_text) :]
        previous_text = current_text

        if delta:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": delta},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {chunk}\n\n"

    # 종료 신호
    yield "data: [DONE]\n\n"
