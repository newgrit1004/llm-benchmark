#!/usr/bin/env python3
"""범용 LLM 추론 성능 벤치마크 스크립트.

이 모듈은 vLLM과 TensorRT-LLM 추론 엔진의 성능을 측정하고 비교하기 위한
벤치마크 도구입니다. OpenAI API 호환 엔드포인트를 통해 두 엔진의 성능을
동일한 조건에서 측정할 수 있습니다.

주요 기능:
    - OpenAI API 호환 엔드포인트 테스팅 (/v1/completions)
    - 워밍업(warm-up) 단계를 통한 정확한 성능 측정
    - 지연시간(latency) 및 처리량(throughput) 통계 수집
    - 백분위수(percentile) 기반 상세 성능 분석
    - JSON 형식의 결과 저장 및 비교 지원

벤치마크 메트릭:
    - 지연시간: 요청부터 응답까지의 시간 (평균, 중앙값, min/max, p50/p90/p95/p99)
    - 처리량: 초당 생성 토큰 수 (tokens/s)
    - 성공률: 전체 요청 대비 성공한 요청 비율
    - 토큰 사용량: 프롬프트 토큰, 생성 토큰 통계

사용 예시:
    # 기본 실행 (환경변수 없이)
    python3 benchmark_llm.py

    # vLLM 벤치마크 (포트 8000)
    BENCHMARK_PORT=8000 BENCHMARK_OUTPUT=results_vllm.json python3 benchmark_llm.py

    # TensorRT-LLM 벤치마크 (포트 8001, 워밍업 5회)
    BENCHMARK_PORT=8001 BENCHMARK_OUTPUT=results_tensorrt.json BENCHMARK_WARMUP=5 python3 benchmark_llm.py

    # 커스텀 URL 사용
    BENCHMARK_URL=http://custom-server:9000/v1/completions python3 benchmark_llm.py

환경 변수:
    BENCHMARK_PORT: API 서버 포트 (기본값: 8001)
    BENCHMARK_URL: 전체 API URL (지정 시 BENCHMARK_PORT 무시)
    BENCHMARK_OUTPUT: 결과 저장 파일명 (기본값: results_tensorrt.json)
    BENCHMARK_WARMUP: 워밍업 요청 횟수 (기본값: 3)

테스트 설정:
    - 모델: Qwen3-8B
    - 요청 횟수: 10회 (워밍업 제외)
    - 입력 길이: ~128 토큰
    - 출력 길이: 128 토큰
    - Temperature: 0.7

워밍업 단계:
    첫 몇 번의 추론 요청은 모델 로딩, 캐시 초기화 등으로 인해
    실제 성능보다 느릴 수 있습니다. 워밍업 단계는 이러한 초기화
    오버헤드를 제거하여 정확한 성능 측정을 가능하게 합니다.

출력 형식:
    실행 중 진행 상황을 실시간으로 출력하며, 완료 후 다음을 포함한
    상세한 통계를 표시합니다:
    - 성공/실패 요청 수
    - 지연시간 통계 (평균, 중앙값, min/max, 백분위수)
    - 처리량 통계 (평균, 중앙값, min/max)
    - 토큰 사용량 통계
    - JSON 파일로 저장된 상세 결과

참고:
    - 벤치마크 실행 전 대상 서버가 실행 중이어야 합니다
    - Makefile의 make benchmark 명령으로 자동화된 벤치마크 가능
    - 결과 파일은 make compare 명령으로 비교 가능
"""
import json
import time
import sys
import os
from statistics import mean, median
import requests

# Test configuration (matching vLLM benchmark)
# API_URL can be overridden by environment variable or command line argument
DEFAULT_PORT = os.environ.get("BENCHMARK_PORT", "8001")
API_URL = os.environ.get("BENCHMARK_URL", f"http://localhost:{DEFAULT_PORT}/v1/completions")
MODEL = "/models/Qwen3-8B"
NUM_REQUESTS = 10
NUM_WARMUP = int(os.environ.get("BENCHMARK_WARMUP", "3"))  # Number of warm-up requests
INPUT_LENGTH = 128
OUTPUT_LENGTH = 128
TEMPERATURE = 0.7

# Test prompt (same as vLLM)
PROMPT = " ".join(["machine learning", "artificial intelligence", "neural network",
                   "deep learning", "computer vision", "natural language processing",
                   "model training", "inference", "optimization", "performance", "benchmark"] * 20)

def run_inference(prompt: str, max_tokens: int) -> dict:
    """단일 추론 요청을 실행하고 성능 메트릭을 반환합니다.

    이 함수는 OpenAI API 호환 엔드포인트로 POST 요청을 보내고,
    응답 시간과 생성된 토큰 수를 측정합니다. 요청 시작부터 응답 수신까지의
    전체 시간을 지연시간(latency)으로 측정하며, 이를 통해 처리량(throughput)을
    계산합니다.

    Args:
        prompt: 완성(completion)을 생성할 입력 텍스트 프롬프트.
            일반적으로 기술 용어를 반복한 긴 텍스트를 사용하여
            일관된 토큰 길이를 유지합니다.
        max_tokens: 생성할 최대 토큰 수. 출력 길이를 제어하는 데 사용됩니다.
            값이 클수록 더 긴 응답이 생성되지만 처리 시간도 증가합니다.

    Returns:
        dict: 추론 결과를 포함하는 딕셔너리. 다음 키를 포함합니다:
            - success (bool): 요청 성공 여부
            - latency (float): 요청부터 응답까지의 시간 (초 단위)
            - prompt_tokens (int): 입력 프롬프트의 토큰 수 (성공 시)
            - completion_tokens (int): 생성된 출력의 토큰 수 (성공 시)
            - total_tokens (int): 총 토큰 수 (prompt + completion, 성공 시)
            - tokens_per_second (float): 초당 생성된 토큰 수 (성공 시)
            - response (str): 생성된 텍스트 응답 (성공 시)
            - error (str): 오류 메시지 (실패 시)

    예시:
        >>> result = run_inference("What is machine learning?", 50)
        >>> if result["success"]:
        ...     print(f"Latency: {result['latency']:.2f}s")
        ...     print(f"Throughput: {result['tokens_per_second']:.2f} tokens/s")
        ...     print(f"Generated: {result['completion_tokens']} tokens")
        Latency: 1.23s
        Throughput: 40.65 tokens/s
        Generated: 50 tokens

        >>> # 실패한 요청의 경우
        >>> failed_result = run_inference("test", 100)
        >>> if not failed_result["success"]:
        ...     print(f"Error: {failed_result['error']}")
        Error: Connection refused

    참고:
        - 타임아웃은 60초로 설정되어 있습니다
        - HTTP 상태 코드가 200이 아니면 실패로 처리됩니다
        - 지연시간이 0인 경우 tokens_per_second는 0으로 계산됩니다
        - 응답 본문은 OpenAI API 형식을 따라야 합니다 (choices, usage 필드 포함)
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False
    }

    start_time = time.time()
    response = requests.post(API_URL, json=payload, timeout=60)
    latency = time.time() - start_time

    if response.status_code != 200:
        return {
            "success": False,
            "error": response.text,
            "latency": latency
        }

    data = response.json()
    choice = data["choices"][0]
    usage = data["usage"]

    return {
        "success": True,
        "latency": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "tokens_per_second": usage.get("completion_tokens", 0) / latency if latency > 0 else 0,
        "response": choice["text"]
    }

def main():
    """벤치마크 메인 실행 함수.

    이 함수는 전체 벤치마크 프로세스를 관리하고 실행합니다.
    워밍업 단계, 실제 벤치마크 실행, 통계 계산, 결과 출력 및 저장의
    전체 워크플로우를 순차적으로 수행합니다.

    실행 단계:
        1. 설정 정보 출력: API URL, 모델, 요청 횟수 등
        2. 워밍업 단계: NUM_WARMUP 횟수만큼 사전 추론 실행
        3. 벤치마크 단계: NUM_REQUESTS 횟수만큼 측정용 추론 실행
        4. 통계 계산: 지연시간, 처리량, 토큰 사용량 통계 산출
        5. 결과 출력: 콘솔에 상세한 통계 정보 표시
        6. 결과 저장: JSON 파일로 상세 결과 저장

    워밍업 단계의 중요성:
        첫 몇 번의 추론은 다음 이유로 실제 성능보다 느릴 수 있습니다:
        - 모델 가중치의 GPU 메모리 로딩
        - CUDA 커널 초기화 및 컴파일
        - KV 캐시 메모리 할당
        - 배치 처리 파이프라인 초기화
        워밍업 단계는 이러한 초기화 오버헤드를 제거하여
        순수한 추론 성능을 측정할 수 있게 합니다.

    통계 계산:
        성공한 요청들에 대해 다음 통계를 계산합니다:
        - 지연시간 통계:
            * 평균(mean), 중앙값(median), 최소(min), 최대(max)
            * 백분위수: p50, p90, p95, p99
        - 처리량 통계:
            * 평균, 중앙값, 최소, 최대 (tokens/s)
        - 토큰 사용량:
            * 평균 프롬프트 토큰 수
            * 평균 생성 토큰 수
            * 총 생성 토큰 수

    환경 변수 사용:
        BENCHMARK_PORT: API 서버 포트 (기본값: 8001)
        BENCHMARK_URL: 전체 API URL (우선순위 높음)
        BENCHMARK_OUTPUT: 결과 파일명 (기본값: results_tensorrt.json)
        BENCHMARK_WARMUP: 워밍업 요청 횟수 (기본값: 3)

    출력 파일 형식:
        JSON 형식으로 다음 정보를 저장합니다:
        {
            "model": "모델 경로",
            "config": {"input_length", "output_length", "temperature"},
            "requests": {"total", "successful", "failed"},
            "latency_seconds": {"mean", "median", "min", "max", "p50", "p90", "p95", "p99"},
            "throughput_tokens_per_sec": {"mean", "median", "min", "max"},
            "tokens": {"avg_prompt_tokens", "avg_completion_tokens", "total_completion_tokens"},
            "raw_results": [개별 요청 결과 배열]
        }

    Returns:
        None: 함수는 값을 반환하지 않으며, 콘솔 출력과 파일 저장으로 결과를 제공합니다.
        모든 요청이 실패한 경우 오류 메시지를 출력하고 조기 종료합니다.

    예시:
        >>> # 환경 변수 설정 후 실행
        >>> import os
        >>> os.environ["BENCHMARK_PORT"] = "8000"
        >>> os.environ["BENCHMARK_OUTPUT"] = "my_results.json"
        >>> main()
        🚀 Starting TensorRT-LLM Benchmark
        📊 Configuration:
           - API: http://localhost:8000/v1/completions
           - Model: /models/Qwen3-8B
           - Warm-up requests: 3
           - Benchmark requests: 10
        ...
        💾 Results saved to my_results.json

    참고:
        - 모든 요청이 실패하면 통계를 계산하지 않고 종료합니다
        - 일부 요청만 실패한 경우 성공한 요청들로만 통계를 계산합니다
        - 백분위수 계산 시 인덱스가 배열 범위를 벗어나지 않도록 int()로 변환합니다
        - 결과 파일은 기존 파일이 있으면 덮어씁니다
    """
    print("🚀 Starting TensorRT-LLM Benchmark")
    print(f"📊 Configuration:")
    print(f"   - API: {API_URL}")
    print(f"   - Model: {MODEL}")
    print(f"   - Warm-up requests: {NUM_WARMUP}")
    print(f"   - Benchmark requests: {NUM_REQUESTS}")
    print(f"   - Input length: ~{INPUT_LENGTH} tokens")
    print(f"   - Output length: {OUTPUT_LENGTH} tokens")
    print(f"   - Temperature: {TEMPERATURE}")
    print()

    # Warm-up phase
    if NUM_WARMUP > 0:
        print(f"🔥 Warm-up phase ({NUM_WARMUP} requests)...")
        warmup_successful = 0
        for i in range(NUM_WARMUP):
            print(f"   Warm-up {i+1}/{NUM_WARMUP}...", end=" ", flush=True)
            result = run_inference(PROMPT, OUTPUT_LENGTH)
            if result["success"]:
                warmup_successful += 1
                print(f"✅ {result['latency']:.2f}s")
            else:
                print(f"❌ Failed")
        print(f"✅ Warm-up complete ({warmup_successful}/{NUM_WARMUP} successful)")
        print()

    # Benchmark phase
    print(f"📊 Benchmark phase ({NUM_REQUESTS} requests)...")
    results = []
    successful = 0
    failed = 0

    for i in range(NUM_REQUESTS):
        print(f"📤 Request {i+1}/{NUM_REQUESTS}...", end=" ", flush=True)
        result = run_inference(PROMPT, OUTPUT_LENGTH)
        results.append(result)

        if result["success"]:
            successful += 1
            print(f"✅ {result['latency']:.2f}s | {result['tokens_per_second']:.2f} tokens/s")
        else:
            failed += 1
            print(f"❌ Error: {result.get('error', 'Unknown')}")

    # Calculate statistics
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print("\n❌ All requests failed!")
        return

    latencies = [r["latency"] for r in successful_results]
    throughputs = [r["tokens_per_second"] for r in successful_results]

    statistics = {
        "model": MODEL,
        "config": {
            "input_length": INPUT_LENGTH,
            "output_length": OUTPUT_LENGTH,
            "temperature": TEMPERATURE
        },
        "requests": {
            "total": NUM_REQUESTS,
            "successful": successful,
            "failed": failed
        },
        "latency_seconds": {
            "mean": mean(latencies),
            "median": median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "p50": sorted(latencies)[len(latencies)//2],
            "p90": sorted(latencies)[int(len(latencies)*0.9)],
            "p95": sorted(latencies)[int(len(latencies)*0.95)],
            "p99": sorted(latencies)[int(len(latencies)*0.99)]
        },
        "throughput_tokens_per_sec": {
            "mean": mean(throughputs),
            "median": median(throughputs),
            "min": min(throughputs),
            "max": max(throughputs)
        },
        "tokens": {
            "avg_prompt_tokens": int(mean([r["prompt_tokens"] for r in successful_results])),
            "avg_completion_tokens": int(mean([r["completion_tokens"] for r in successful_results])),
            "total_completion_tokens": sum(r["completion_tokens"] for r in successful_results)
        },
        "raw_results": results
    }

    # Print summary
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    print(f"✅ Successful: {successful}/{NUM_REQUESTS}")
    print(f"❌ Failed: {failed}/{NUM_REQUESTS}")
    print()
    print("⏱️  Latency:")
    print(f"   - Mean: {statistics['latency_seconds']['mean']:.3f}s")
    print(f"   - Median: {statistics['latency_seconds']['median']:.3f}s")
    print(f"   - Min: {statistics['latency_seconds']['min']:.3f}s")
    print(f"   - Max: {statistics['latency_seconds']['max']:.3f}s")
    print()
    print("🚀 Throughput:")
    print(f"   - Mean: {statistics['throughput_tokens_per_sec']['mean']:.2f} tokens/s")
    print(f"   - Median: {statistics['throughput_tokens_per_sec']['median']:.2f} tokens/s")
    print(f"   - Min: {statistics['throughput_tokens_per_sec']['min']:.2f} tokens/s")
    print(f"   - Max: {statistics['throughput_tokens_per_sec']['max']:.2f} tokens/s")
    print()
    print("📝 Tokens:")
    print(f"   - Avg prompt: {statistics['tokens']['avg_prompt_tokens']}")
    print(f"   - Avg completion: {statistics['tokens']['avg_completion_tokens']}")
    print(f"   - Total completion: {statistics['tokens']['total_completion_tokens']}")
    print("="*60)

    # Save results
    output_file = os.environ.get("BENCHMARK_OUTPUT", "results_tensorrt.json")
    with open(output_file, "w") as f:
        json.dump(statistics, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")

if __name__ == "__main__":
    main()
