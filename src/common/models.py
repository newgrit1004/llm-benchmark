"""API 요청 및 응답을 위한 공유 Pydantic 모델.

이 모듈은 OpenAI API 사양을 따르는 일관된 API 인터페이스를 보장하기 위해
vLLM 및 TensorRT-LLM 서버 모두에서 사용되는 데이터 모델을 정의합니다.

모든 모델은 Pydantic을 사용하여 자동 검증, 직렬화 및 문서화를 제공합니다.
이 모델들은 /v1/completions 및 /v1/chat/completions 엔드포인트를 모두 지원합니다.

사용 예시:
    from src.common.models import CompletionRequest, ChatCompletionRequest

    # Completion 요청 생성
    request = CompletionRequest(
        model="/models/Qwen3-8B",
        prompt="안녕하세요!",
        max_tokens=100,
        temperature=0.7
    )

    # Chat Completion 요청 생성
    chat_request = ChatCompletionRequest(
        model="/models/Qwen3-8B",
        messages=[
            ChatMessage(role="user", content="안녕하세요!")
        ],
        max_tokens=100
    )
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """채팅 대화의 단일 메시지를 나타냅니다.

    이 모델은 각 메시지가 역할(role)과 내용(content)을 갖는
    OpenAI Chat API 형식을 따릅니다.

    속성:
        role: 메시지 발신자의 역할. 유효한 값:
            - "system": 시스템 지시사항 또는 컨텍스트
            - "user": 사용자의 입력 메시지
            - "assistant": AI 어시스턴트의 응답
        content: 메시지의 실제 텍스트 내용. 임의의 문자열 가능.

    예시:
        >>> msg = ChatMessage(role="user", content="머신러닝이 뭔가요?")
        >>> print(msg.role, msg.content)
        user 머신러닝이 뭔가요?

        >>> system_msg = ChatMessage(
        ...     role="system",
        ...     content="당신은 도움이 되는 AI 어시스턴트입니다."
        ... )
    """

    role: str = Field(description="메시지 역할 (system, user, assistant)")
    content: str = Field(description="메시지 내용")


class CompletionRequest(BaseModel):
    """/v1/completions 엔드포인트를 위한 요청 모델.

    이 모델은 OpenAI API와 호환되는 텍스트 완성 요청을 나타냅니다.
    프롬프트를 받아 다양한 매개변수를 기반으로 연속된 텍스트를 생성합니다.

    속성:
        model: 완성에 사용할 모델의 경로 또는 식별자.
            예시: "/models/Qwen3-8B"
        prompt: 완성할 입력 텍스트. 다음 중 하나:
            - 단일 문자열: "옛날 옛적에"
            - 문자열 리스트: ["안녕하세요", "어떻게 지내세요"]
        max_tokens: 생성할 최대 토큰 수.
            범위: 1-4096. 기본값: 16.
            참고: 일부 구현에서는 프롬프트와 완성 토큰을 모두 포함합니다.
        temperature: 생성 시 무작위성을 제어합니다.
            범위: 0.0-2.0. 기본값: 1.0.
            - 낮음 (0.0): 더 결정적이고 집중된 출력
            - 높음 (2.0): 더 무작위적이고 창의적인 출력
        top_p: Nucleus sampling 매개변수.
            범위: 0.0-1.0. 기본값: 1.0.
            상위 P 확률 질량에서 샘플링하여 다양성을 제어합니다.
            예시: 0.9는 상위 90% 확률 토큰에서 샘플링함을 의미합니다.
        n: 생성할 완성의 수.
            범위: 1-5. 기본값: 1.
            동일한 프롬프트에 대해 여러 개의 독립적인 완성을 생성합니다.
        stream: 스트리밍 응답 활성화.
            기본값: False.
            - True: 생성되는 대로 토큰 반환 (SSE 형식)
            - False: 생성 완료 후 전체 응답 반환
        stop: 생성을 중단할 시퀀스.
            다음 중 하나:
            - 단일 문자열: "\\n"
            - 문자열 리스트: ["\\n", ".", "!"]
            - None: 중단 시퀀스 없음 (max_tokens까지 생성)
        presence_penalty: 프롬프트에 나타난 토큰 사용에 대한 패널티.
            범위: -2.0~2.0. 기본값: 0.0.
            - 양수: 새로운 주제 논의 권장
            - 음수: 프롬프트 주제 반복 허용
        frequency_penalty: 생성된 텍스트에서 토큰 빈도에 기반한 패널티.
            범위: -2.0~2.0. 기본값: 0.0.
            - 양수: 동일한 토큰의 반복 감소
            - 음수: 더 많은 반복 허용

    예시:
        >>> request = CompletionRequest(
        ...     model="/models/Qwen3-8B",
        ...     prompt="AI의 미래는",
        ...     max_tokens=100,
        ...     temperature=0.7,
        ...     top_p=0.9,
        ...     stop=["\\n", "."]
        ... )

        >>> # 스트리밍 요청
        >>> stream_request = CompletionRequest(
        ...     model="/models/Qwen3-8B",
        ...     prompt="양자 컴퓨팅을 설명하세요:",
        ...     max_tokens=200,
        ...     stream=True
        ... )

        >>> # 다중 완성
        >>> multi_request = CompletionRequest(
        ...     model="/models/Qwen3-8B",
        ...     prompt="하이쿠를 작성하세요",
        ...     n=3,
        ...     max_tokens=50
        ... )
    """

    model: str
    prompt: str | list[str]
    max_tokens: int = Field(default=16, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=5)
    stream: bool = False
    stop: Optional[str | list[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class ChatCompletionRequest(BaseModel):
    """/v1/chat/completions 엔드포인트를 위한 요청 모델.

    이 모델은 OpenAI Chat API와 호환되는 채팅 완성 요청을 나타냅니다.
    대화 기록(메시지 리스트)을 받아 컨텍스트를 기반으로
    다음 어시스턴트 응답을 생성합니다.

    채팅 형식의 장점:
    - 컨텍스트가 있는 다중 턴 대화
    - 동작을 안내하는 시스템 지시사항
    - 역할 기반 메시지 포매팅

    속성:
        model: 채팅 완성에 사용할 모델의 경로 또는 식별자.
            예시: "/models/Qwen3-8B"
        messages: ChatMessage 객체의 리스트로 표현되는 대화 기록.
            각 메시지는 역할과 내용을 가져야 합니다.
            일반적인 순서: [system, user, assistant, user, ...]
        max_tokens: 응답에서 생성할 최대 토큰 수.
            범위: 1-4096. 기본값: 16.
            이는 어시스턴트의 응답만 제한하며, 전체 컨텍스트는 아닙니다.
        temperature: 어시스턴트 응답의 무작위성을 제어합니다.
            범위: 0.0-2.0. 기본값: 1.0.
            - 0.0: 결정적이고 일관된 응답
            - 1.0: 창의성과 일관성의 균형
            - 2.0: 매우 창의적이고 예측 불가능
        top_p: 응답 생성을 위한 Nucleus sampling.
            범위: 0.0-1.0. 기본값: 1.0.
            토큰 샘플링 풀을 제한하여 다양성을 제어합니다.
        n: 생성할 채팅 완성 응답의 수.
            범위: 1-5. 기본값: 1.
            각 완성은 동일한 메시지 기록이 주어졌을 때 독립적입니다.
        stream: 어시스턴트 응답의 스트리밍 활성화.
            기본값: False.
            - True: 생성되는 대로 토큰 스트리밍 (Server-Sent Events)
            - False: 생성 완료 후 전체 응답 반환
        stop: 생성을 중단할 토큰 시퀀스.
            문자열, 문자열 리스트, 또는 None 가능.
            예시: ["\\n\\n", "User:", "Assistant:"]

    예시:
        >>> # 간단한 질의응답
        >>> request = ChatCompletionRequest(
        ...     model="/models/Qwen3-8B",
        ...     messages=[
        ...         ChatMessage(role="user", content="파이썬이 뭔가요?")
        ...     ],
        ...     max_tokens=100
        ... )

        >>> # 시스템 지시사항 포함
        >>> request = ChatCompletionRequest(
        ...     model="/models/Qwen3-8B",
        ...     messages=[
        ...         ChatMessage(
        ...             role="system",
        ...             content="당신은 도움이 되는 코딩 어시스턴트입니다."
        ...         ),
        ...         ChatMessage(
        ...             role="user",
        ...             content="파이썬에서 리스트를 어떻게 뒤집나요?"
        ...         )
        ...     ],
        ...     max_tokens=150,
        ...     temperature=0.7
        ... )

        >>> # 다중 턴 대화
        >>> request = ChatCompletionRequest(
        ...     model="/models/Qwen3-8B",
        ...     messages=[
        ...         ChatMessage(role="system", content="당신은 수학 선생님입니다."),
        ...         ChatMessage(role="user", content="15 + 27은 얼마인가요?"),
        ...         ChatMessage(role="assistant", content="15 + 27은 42입니다."),
        ...         ChatMessage(role="user", content="그럼 42 * 2는요?")
        ...     ],
        ...     max_tokens=50
        ... )

        >>> # 스트리밍 채팅
        >>> stream_request = ChatCompletionRequest(
        ...     model="/models/Qwen3-8B",
        ...     messages=[
        ...         ChatMessage(
        ...             role="user",
        ...             content="양자 얽힘을 설명해주세요."
        ...         )
        ...     ],
        ...     stream=True,
        ...     max_tokens=300
        ... )

    참고:
        - 모델은 전체 대화 기록을 컨텍스트로 사용합니다
        - 긴 대화는 더 많은 토큰과 메모리를 소비합니다
        - 시스템 메시지는 일반적으로 동작을 설정하기 위해 처음에 나타납니다
        - Temperature와 top_p는 함께 작동하므로 한 번에 하나씩 조정하세요
        - 스트리밍은 실시간 사용자 경험에 유용합니다
    """

    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=16, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=5)
    stream: bool = False
    stop: Optional[str | list[str]] = None
