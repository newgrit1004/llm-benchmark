"""공유 유틸리티 함수 모음.

이 모듈은 vLLM 및 TensorRT-LLM 서버에서 공통으로 사용하는 유틸리티 함수를 제공합니다.
주로 채팅 메시지 처리와 프롬프트 포맷팅 기능을 포함합니다.

사용 예시:
    from src.common.utils import format_chat_prompt
    from src.common.models import ChatMessage

    # 채팅 메시지를 프롬프트로 변환
    messages = [
        ChatMessage(role="system", content="당신은 도움이 되는 AI입니다."),
        ChatMessage(role="user", content="안녕하세요!")
    ]
    prompt = format_chat_prompt(messages)
"""

from .models import ChatMessage


def format_chat_prompt(messages: list[ChatMessage]) -> str:
    """채팅 메시지 리스트를 텍스트 프롬프트 형식으로 변환합니다.

    이 함수는 OpenAI Chat API 형식의 메시지 리스트를 받아서
    텍스트 기반 LLM이 이해할 수 있는 프롬프트 문자열로 변환합니다.
    각 메시지는 "역할: 내용" 형식으로 포맷되며, 마지막에 "Assistant:" 접두사를
    추가하여 LLM이 응답을 생성하도록 유도합니다.

    Args:
        messages: ChatMessage 객체의 리스트. 각 메시지는 role과 content를 가집니다.
            role은 "system", "user", "assistant" 중 하나여야 합니다.

    Returns:
        str: 포맷된 프롬프트 문자열. 각 메시지는 새 줄로 구분되며,
            마지막에 "Assistant:"가 추가됩니다.

    예시:
        >>> from src.common.models import ChatMessage
        >>> messages = [
        ...     ChatMessage(role="system", content="당신은 도움이 되는 AI입니다."),
        ...     ChatMessage(role="user", content="파이썬이 뭔가요?")
        ... ]
        >>> print(format_chat_prompt(messages))
        System: 당신은 도움이 되는 AI입니다.
        User: 파이썬이 뭔가요?
        Assistant:

        >>> # 다중 턴 대화
        >>> messages = [
        ...     ChatMessage(role="system", content="수학 선생님입니다."),
        ...     ChatMessage(role="user", content="2 + 2는?"),
        ...     ChatMessage(role="assistant", content="4입니다."),
        ...     ChatMessage(role="user", content="10 * 5는?")
        ... ]
        >>> print(format_chat_prompt(messages))
        System: 수학 선생님입니다.
        User: 2 + 2는?
        Assistant: 4입니다.
        User: 10 * 5는?
        Assistant:

    참고:
        - 반환되는 프롬프트는 항상 "Assistant:"로 끝나며, 이는 LLM이
          어시스턴트 역할로 다음 응답을 생성하도록 유도합니다
        - system, user, assistant 외의 role은 무시됩니다
        - 메시지 순서는 입력 리스트의 순서를 그대로 유지합니다
        - 빈 메시지 리스트를 전달하면 "Assistant:"만 반환됩니다
    """
    formatted = []
    for msg in messages:
        if msg.role == "system":
            formatted.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            formatted.append(f"Assistant: {msg.content}")

    formatted.append("Assistant:")
    return "\n".join(formatted)
