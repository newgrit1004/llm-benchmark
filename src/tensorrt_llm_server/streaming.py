"""Streaming response generators for TensorRT-LLM."""

import asyncio
import json
import time
from typing import AsyncIterator

from common.logger import get_logger
from .engine import get_llm

logger = get_logger(__name__)


async def stream_completion(
    prompt: str, sampling_params, request_id: str
) -> AsyncIterator[str]:
    """스트리밍 completion 생성"""
    llm = get_llm()

    # TensorRT-LLM 스트리밍은 구현에 따라 다를 수 있음
    # 여기서는 간단한 구현 예시
    try:
        # 전체 텍스트 생성
        outputs = llm.generate([prompt], sampling_params)

        if hasattr(outputs[0], "outputs"):
            text = outputs[0].outputs[0].text
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])

        # 청크로 나눠서 전송
        chunk_size = max(1, len(text) // 10)
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]

            data = {
                "id": request_id,
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "choices": [
                    {
                        "text": chunk,
                        "index": 0,
                        "finish_reason": None if i + chunk_size < len(text) else "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.01)  # 스트리밍 효과

        # 종료 신호
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f'data: {{"error": "{str(e)}"}}\n\n'


async def stream_chat_completion(
    prompt: str, sampling_params, request_id: str
) -> AsyncIterator[str]:
    """스트리밍 chat completion 생성"""
    llm = get_llm()

    try:
        # 전체 텍스트 생성
        outputs = llm.generate([prompt], sampling_params)

        if hasattr(outputs[0], "outputs"):
            text = outputs[0].outputs[0].text
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])

        # 청크로 나눠서 전송
        chunk_size = max(1, len(text) // 10)
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]

            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": chunk},
                        "finish_reason": None if i + chunk_size < len(text) else "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.01)  # 스트리밍 효과

        # 종료 신호
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f'data: {{"error": "{str(e)}"}}\n\n'
