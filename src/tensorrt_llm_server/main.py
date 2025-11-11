#!/usr/bin/env python3
"""TensorRT-LLM 추론 서버 메인 엔트리포인트.

이 모듈은 TensorRT-LLM을 사용하는 OpenAI 호환 API 서버를 시작합니다.
FastAPI 기반으로 동작하며, /v1/completions 및 /v1/chat/completions 엔드포인트를 제공합니다.

주요 기능:
    - OpenAI API 호환 인터페이스
    - TensorRT-LLM 최적화 엔진 사용
    - Tensor parallelism 지원
    - 스트리밍 응답 지원
    - 사전 컴파일된 엔진 로드

사용 예시:
    # 기본 실행
    python3 -m src.tensorrt_llm_server.main \\
        --engine-dir /engines/qwen3-8b/engine

    # 고급 설정
    python3 -m src.tensorrt_llm_server.main \\
        --engine-dir /engines/qwen3-8b/engine \\
        --model-name "Qwen3-8B" \\
        --tensor-parallel-size 2 \\
        --host 0.0.0.0 \\
        --port 8001

커맨드라인 인자:
    --engine-dir: TensorRT-LLM 엔진 디렉토리 경로 (필수)
    --model-name: 모델 이름 (선택사항, API 응답에 사용)
    --tensor-parallel-size: Tensor parallelism 크기 (기본값: 1)
    --host: 서버 호스트 (기본값: 0.0.0.0)
    --port: 서버 포트 (기본값: 8001)

참고:
    - TensorRT-LLM 엔진은 사전에 빌드되어 있어야 합니다
    - 엔진 빌드는 scripts/tensorrt_startup.sh에서 자동으로 수행됩니다
    - 첫 실행 시 모델 변환 및 엔진 빌드로 20-30분 소요될 수 있습니다
"""

import argparse
import uvicorn
from common.logger import setup_logger
from .engine import init_engine
from .routes import app

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM Custom Inference Server"
    )

    # 모델 설정
    parser.add_argument(
        "--engine-dir", type=str, required=True, help="TensorRT-LLM 엔진 디렉토리"
    )
    parser.add_argument("--model-name", type=str, default=None, help="모델 이름 (선택사항)")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Tensor parallelism 크기"
    )

    # 서버 설정
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8001, help="서버 포트")

    args = parser.parse_args()

    # 엔진 초기화
    init_engine(args)

    # 서버 시작
    logger.info(f"Starting TensorRT-LLM server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
