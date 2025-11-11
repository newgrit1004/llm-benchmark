#!/usr/bin/env python3
"""vLLM 추론 서버 메인 엔트리포인트.

이 모듈은 vLLM을 사용하는 OpenAI 호환 API 서버를 시작합니다.
FastAPI 기반으로 동작하며, /v1/completions 및 /v1/chat/completions 엔드포인트를 제공합니다.

주요 기능:
    - OpenAI API 호환 인터페이스
    - 커스텀 vLLM 엔진 설정 지원
    - Tensor parallelism 지원
    - GPU 메모리 사용률 조정
    - 스트리밍 응답 지원

사용 예시:
    # 기본 실행
    python3 -m src.vllm_server.main --model /models/Qwen3-8B

    # 고급 설정
    python3 -m src.vllm_server.main \\
        --model /models/Qwen3-8B \\
        --tensor-parallel-size 2 \\
        --gpu-memory-utilization 0.8 \\
        --max-model-len 4096 \\
        --host 0.0.0.0 \\
        --port 8000

커맨드라인 인자:
    --model: 모델 경로 또는 HuggingFace 모델 ID (필수)
    --tensor-parallel-size: Tensor parallelism 크기 (기본값: 1)
    --dtype: 모델 데이터 타입 - float16, bfloat16 (기본값: bfloat16)
    --gpu-memory-utilization: GPU 메모리 사용률 0.0~1.0 (기본값: 0.9)
    --max-model-len: 최대 시퀀스 길이 (기본값: 8192)
    --enforce-eager: Eager 모드 강제 (디버깅용)
    --host: 서버 호스트 (기본값: 0.0.0.0)
    --port: 서버 포트 (기본값: 8000)
"""

import argparse
import asyncio
import uvicorn
from common.logger import setup_logger
from .engine import init_engine
from .routes import app

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="vLLM Custom Inference Server")

    # 모델 설정
    parser.add_argument(
        "--model", type=str, required=True, help="모델 경로 또는 HuggingFace 모델 ID"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Tensor parallelism 크기"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="모델 dtype",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9, help="GPU 메모리 사용률"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192, help="최대 시퀀스 길이"
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Eager mode 강제 (디버깅용)"
    )

    # 서버 설정
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")

    args = parser.parse_args()

    # 엔진 초기화
    asyncio.run(init_engine(args))

    # 서버 시작
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
