"""TensorRT-LLM engine initialization and management."""

import logging
from typing import Optional, Any
from pathlib import Path

from common.logger import get_logger

# TensorRT-LLM imports
try:
    from tensorrt_llm.hlapi import LLM, SamplingParams

    TENSORRT_AVAILABLE = True
except ImportError:
    logging.warning("TensorRT-LLM not found. Using mock implementation.")
    LLM = None
    SamplingParams = None
    TENSORRT_AVAILABLE = False

logger = get_logger(__name__)

# 전역 변수
_llm: Optional[Any] = None
_model_config: dict = {}


def init_engine(args) -> None:
    """엔진 초기화"""
    global _llm, _model_config

    logger.info("Initializing TensorRT-LLM engine...")
    logger.info(f"Engine directory: {args.engine_dir}")

    try:
        if TENSORRT_AVAILABLE and LLM:
            # TensorRT-LLM HLAPI 사용
            _llm = LLM(
                model=args.engine_dir,
                tensor_parallel_size=args.tensor_parallel_size,
            )

            _model_config = {
                "model_name": args.model_name or Path(args.engine_dir).name,
                "engine_dir": args.engine_dir,
                "tensor_parallel_size": args.tensor_parallel_size,
            }

            logger.info("TensorRT-LLM engine initialized successfully!")
        else:
            logger.warning("TensorRT-LLM not available. Running in mock mode.")
            _model_config = {
                "model_name": "mock-model",
                "engine_dir": args.engine_dir,
            }

    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise


def get_llm() -> Optional[Any]:
    """LLM 인스턴스 반환"""
    return _llm


def get_model_config() -> dict:
    """모델 설정 반환"""
    return _model_config


def get_sampling_params_class():
    """SamplingParams 클래스 반환"""
    return SamplingParams
