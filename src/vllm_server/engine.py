"""vLLM engine initialization and management."""

from typing import Optional
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from common.logger import get_logger

logger = get_logger(__name__)

# 전역 엔진 변수
_engine: Optional[AsyncLLMEngine] = None


async def init_engine(args) -> None:
    """엔진 초기화"""
    global _engine

    logger.info("Initializing vLLM engine...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    logger.info(f"GPU Memory Utilization: {args.gpu_memory_utilization}")

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )

    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("vLLM engine initialized successfully!")


def get_engine() -> Optional[AsyncLLMEngine]:
    """엔진 인스턴스 반환"""
    return _engine
