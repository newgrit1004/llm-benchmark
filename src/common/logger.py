"""로깅 설정 및 로거 관리 유틸리티.

이 모듈은 vLLM 및 TensorRT-LLM 서버에서 사용하는 로거 설정을 제공합니다.
표준 Python logging 모듈을 기반으로 일관된 로그 포맷과 레벨을 제공합니다.

사용 예시:
    from src.common.logger import setup_logger, get_logger

    # 서버 시작 시 로거 설정
    logger = setup_logger("vllm_server", level=logging.INFO)
    logger.info("서버 시작 중...")

    # 다른 모듈에서 로거 가져오기
    logger = get_logger("vllm_server")
    logger.debug("디버그 메시지")
"""

import logging


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """로깅 시스템을 설정하고 설정된 로거를 반환합니다.

    이 함수는 로깅 시스템의 기본 설정을 초기화하고 지정된 이름의 로거를 반환합니다.
    로그 메시지는 타임스탬프, 로거 이름, 레벨, 메시지를 포함하는 표준 포맷으로 출력됩니다.

    Args:
        name: 로거 이름. 일반적으로 모듈 이름이나 서버 이름을 사용합니다.
            예: "vllm_server", "tensorrt_llm_server"
        level: 로그 레벨. Python logging 모듈의 상수를 사용합니다.
            기본값: logging.INFO
            가능한 값:
            - logging.DEBUG (10): 상세한 디버그 정보
            - logging.INFO (20): 일반 정보 메시지
            - logging.WARNING (30): 경고 메시지
            - logging.ERROR (40): 오류 메시지
            - logging.CRITICAL (50): 치명적 오류

    Returns:
        logging.Logger: 설정된 로거 객체

    예시:
        >>> import logging
        >>> logger = setup_logger("my_app", level=logging.DEBUG)
        >>> logger.info("애플리케이션 시작")
        2025-01-11 12:00:00,000 - my_app - INFO - 애플리케이션 시작

        >>> logger.debug("디버그 메시지")
        2025-01-11 12:00:01,000 - my_app - DEBUG - 디버그 메시지

        >>> logger.warning("경고 메시지")
        2025-01-11 12:00:02,000 - my_app - WARNING - 경고 메시지

    참고:
        - 이 함수는 전역 로깅 설정을 변경하므로 애플리케이션 시작 시 한 번만 호출해야 합니다
        - 동일한 이름으로 여러 번 호출하면 같은 로거 인스턴스가 반환됩니다
        - 로그 포맷: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """기존에 설정된 로거를 반환합니다.

    이 함수는 이미 생성된 로거를 가져옵니다. setup_logger()로 먼저 로거를 설정한 후
    다른 모듈에서 동일한 로거를 사용할 때 이 함수를 호출합니다.

    Args:
        name: 가져올 로거의 이름. setup_logger()에서 사용한 이름과 동일해야 합니다.

    Returns:
        logging.Logger: 지정된 이름의 로거 객체

    예시:
        >>> # main.py에서 로거 설정
        >>> from src.common.logger import setup_logger
        >>> logger = setup_logger("my_app")

        >>> # routes.py에서 동일한 로거 사용
        >>> from src.common.logger import get_logger
        >>> logger = get_logger("my_app")
        >>> logger.info("API 호출 처리 중")

    참고:
        - 로거가 아직 설정되지 않은 경우에도 로거 객체는 반환되지만
          로그 레벨과 포맷은 기본값으로 설정됩니다
        - 일관된 로깅을 위해 setup_logger()를 먼저 호출하는 것을 권장합니다
    """
    return logging.getLogger(name)
