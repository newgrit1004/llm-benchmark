"""FastAPI routes for vLLM server."""

import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from vllm import SamplingParams
from vllm.utils import random_uuid

from common.models import CompletionRequest, ChatCompletionRequest
from common.utils import format_chat_prompt
from .engine import get_engine
from .streaming import stream_completion, stream_chat_completion

# FastAPI 앱
app = FastAPI(title="vLLM Inference Server", version="1.0.0")


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    engine = get_engine()
    return {"status": "healthy", "engine": "vllm", "model_loaded": engine is not None}


@app.get("/v1/models")
async def list_models():
    """사용 가능한 모델 목록"""
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    return {
        "object": "list",
        "data": [
            {
                "id": engine.engine.model_config.model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm",
            }
        ],
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Completion API 엔드포인트"""
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Sampling parameters 설정
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        n=request.n,
        stop=request.stop if request.stop else None,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
    )

    request_id = random_uuid()

    if request.stream:
        # 스트리밍 응답
        return StreamingResponse(
            stream_completion(request.prompt, sampling_params, request_id),
            media_type="text/event-stream",
        )
    else:
        # 일반 응답
        results = []
        async for output in engine.generate(request.prompt, sampling_params, request_id):
            results.append(output)

        final_output = results[-1]

        return JSONResponse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "text": output.outputs[0].text,
                        "index": i,
                        "logprobs": None,
                        "finish_reason": output.outputs[0].finish_reason,
                    }
                    for i, output in enumerate(final_output.outputs)
                ],
                "usage": {
                    "prompt_tokens": len(final_output.prompt_token_ids),
                    "completion_tokens": len(final_output.outputs[0].token_ids),
                    "total_tokens": len(final_output.prompt_token_ids)
                    + len(final_output.outputs[0].token_ids),
                },
            }
        )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Chat Completion API 엔드포인트"""
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # 메시지를 프롬프트로 변환
    prompt = format_chat_prompt(request.messages)

    # Sampling parameters 설정
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        n=request.n,
        stop=request.stop if request.stop else None,
    )

    request_id = random_uuid()

    if request.stream:
        # 스트리밍 응답
        return StreamingResponse(
            stream_chat_completion(prompt, sampling_params, request_id),
            media_type="text/event-stream",
        )
    else:
        # 일반 응답
        results = []
        async for output in engine.generate(prompt, sampling_params, request_id):
            results.append(output)

        final_output = results[-1]

        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_output.outputs[0].text,
                        },
                        "finish_reason": final_output.outputs[0].finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(final_output.prompt_token_ids),
                    "completion_tokens": len(final_output.outputs[0].token_ids),
                    "total_tokens": len(final_output.prompt_token_ids)
                    + len(final_output.outputs[0].token_ids),
                },
            }
        )
