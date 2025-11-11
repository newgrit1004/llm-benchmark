"""FastAPI routes for TensorRT-LLM server."""

import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from common.models import CompletionRequest, ChatCompletionRequest
from common.utils import format_chat_prompt
from common.logger import get_logger
from .engine import get_llm, get_model_config, get_sampling_params_class
from .streaming import stream_completion, stream_chat_completion

logger = get_logger(__name__)

# FastAPI 앱
app = FastAPI(title="TensorRT-LLM Inference Server", version="1.0.0")


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    llm = get_llm()
    return {
        "status": "healthy",
        "engine": "tensorrt-llm",
        "model_loaded": llm is not None,
    }


@app.get("/v1/models")
async def list_models():
    """사용 가능한 모델 목록"""
    llm = get_llm()
    if llm is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    model_config = get_model_config()
    return {
        "object": "list",
        "data": [
            {
                "id": model_config.get("model_name", "unknown"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tensorrt-llm",
            }
        ],
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Completion API 엔드포인트"""
    llm = get_llm()
    if llm is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Sampling parameters 설정
    SamplingParams = get_sampling_params_class()
    if SamplingParams:
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    else:
        sampling_params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

    request_id = f"cmpl-{int(time.time())}"

    if request.stream:
        # 스트리밍 응답
        return StreamingResponse(
            stream_completion(request.prompt, sampling_params, request_id),
            media_type="text/event-stream",
        )
    else:
        # 일반 응답
        try:
            # TensorRT-LLM으로 생성
            outputs = llm.generate(
                [request.prompt] if isinstance(request.prompt, str) else request.prompt,
                sampling_params,
            )

            # 결과 포맷팅
            choices = []
            for i, output in enumerate(outputs):
                if hasattr(output, "outputs"):
                    text = output.outputs[0].text
                elif isinstance(output, str):
                    text = output
                else:
                    text = str(output)

                choices.append(
                    {
                        "text": text,
                        "index": i,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                )

            return JSONResponse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": choices,
                    "usage": {
                        "prompt_tokens": len(request.prompt.split()),
                        "completion_tokens": sum(
                            len(c["text"].split()) for c in choices
                        ),
                        "total_tokens": len(request.prompt.split())
                        + sum(len(c["text"].split()) for c in choices),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Chat Completion API 엔드포인트"""
    llm = get_llm()
    if llm is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # 메시지를 프롬프트로 변환
    prompt = format_chat_prompt(request.messages)

    # Sampling parameters 설정
    SamplingParams = get_sampling_params_class()
    if SamplingParams:
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    else:
        sampling_params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

    request_id = f"chatcmpl-{int(time.time())}"

    if request.stream:
        # 스트리밍 응답
        return StreamingResponse(
            stream_chat_completion(prompt, sampling_params, request_id),
            media_type="text/event-stream",
        )
    else:
        # 일반 응답
        try:
            outputs = llm.generate([prompt], sampling_params)

            # 결과 포맷팅
            if hasattr(outputs[0], "outputs"):
                text = outputs[0].outputs[0].text
            elif isinstance(outputs[0], str):
                text = outputs[0]
            else:
                text = str(outputs[0])

            return JSONResponse(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(text.split()),
                        "total_tokens": len(prompt.split()) + len(text.split()),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
