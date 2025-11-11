DIR?=.

# ============================================
# 개발 환경 설정
# ============================================

# 프로젝트 초기 설정
.PHONY: setup
setup:
	git config commit.template .gitmessage.txt
	uv sync --all-extras --dev
	uv run pre-commit install

# 의존성만 설치 (프로덕션 + 개발)
.PHONY: install
install:
	uv sync

# 의존성 업데이트 (lock 파일 갱신)
.PHONY: update
update:
	uv lock --upgrade

# ============================================
# 코드 품질 관리
# ============================================

# 코드 포맷팅 (Black + isort 대체)
.PHONY: format
format:
	uv run ruff format ${DIR}

# 코드 린팅 체크
.PHONY: lint
lint:
	uv run ruff check ${DIR}

# 코드 린팅 + 자동 수정
.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix ${DIR}

# 타입 체킹 (mypy/pyright 대체)
.PHONY: typecheck
typecheck:
	uv run ty check

# pre-commit 전체 실행
.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files

# 캐시 및 임시 파일 정리
.PHONY: clean
clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ============================================
# LLM 벤치마크 명령어
# ============================================

# 벤치마크용 로컬 가상환경 설정
.PHONY: setup-benchmark
setup-benchmark:
	@echo "🔧 벤치마크용 로컬 가상환경 설정 중..."
	@if ! command -v uv &> /dev/null; then \
		echo "❌ uv가 설치되어 있지 않습니다."; \
		echo "설치: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	uv venv .venv
	uv pip install --python .venv/bin/python openai python-dotenv
	@echo "✅ 벤치마크 환경 설정 완료!"

# ============================================
# Docker 관리
# ============================================

# Docker 이미지 빌드
.PHONY: build-vllm build-tensorrt build-all
build-vllm:
	@echo "🐳 vLLM 도커 이미지 빌드 중..."
	cd docker_composes && docker-compose -f docker-compose.vllm.yml build
	@echo "✅ vLLM 이미지 빌드 완료!"

build-tensorrt:
	@echo "🐳 TensorRT-LLM 도커 이미지 빌드 중..."
	cd docker_composes && docker-compose -f docker-compose.tensorrt-llm.yml build
	@echo "✅ TensorRT-LLM 이미지 빌드 완료!"

build-all: build-vllm build-tensorrt

# Docker 컨테이너 실행/중지
.PHONY: up-vllm up-tensorrt down-vllm down-tensorrt
up-vllm:
	@echo "🚀 vLLM 컨테이너 시작..."
	cd docker_composes && docker-compose -f docker-compose.vllm.yml up -d
	@echo "✅ API: http://localhost:8000"

up-tensorrt:
	@echo "🚀 TensorRT-LLM 컨테이너 시작..."
	cd docker_composes && docker-compose -f docker-compose.tensorrt-llm.yml up -d
	@echo "✅ API: http://localhost:8001"

down-vllm:
	cd docker_composes && docker-compose -f docker-compose.vllm.yml down

down-tensorrt:
	cd docker_composes && docker-compose -f docker-compose.tensorrt-llm.yml down

# 로그 확인
.PHONY: logs-vllm logs-tensorrt
logs-vllm:
	cd docker_composes && docker-compose -f docker-compose.vllm.yml logs -f

logs-tensorrt:
	cd docker_composes && docker-compose -f docker-compose.tensorrt-llm.yml logs -f

# ============================================
# 벤치마크 실행
# ============================================

# 수동 벤치마크 (컨테이너를 직접 시작한 후 실행)
.PHONY: benchmark-manual-vllm benchmark-manual-tensorrt
benchmark-manual-vllm:
	@echo "📈 vLLM 수동 벤치마크 실행 중..."
	@echo "⚠️  vLLM 컨테이너가 이미 실행 중이어야 합니다 (make up-vllm)"
	@BENCHMARK_PORT=8000 BENCHMARK_OUTPUT=results_vllm_manual.json python3 benchmark_llm.py

benchmark-manual-tensorrt:
	@echo "📈 TensorRT-LLM 수동 벤치마크 실행 중..."
	@echo "⚠️  TensorRT-LLM 컨테이너가 이미 실행 중이어야 합니다 (make up-tensorrt)"
	@BENCHMARK_PORT=8001 BENCHMARK_OUTPUT=results_tensorrt_manual.json python3 benchmark_llm.py

# 순차적 자동 벤치마크 (컨테이너 자동 시작/종료)
.PHONY: benchmark benchmark-auto
benchmark: benchmark-auto
benchmark-auto:
	@echo "============================================"
	@echo "🚀 자동 벤치마크 시작"
	@echo "============================================"
	@echo ""
	@echo "📋 실행 계획:"
	@echo "  1. vLLM 벤치마크 (포트 8000)"
	@echo "  2. TensorRT-LLM 벤치마크 (포트 8001)"
	@echo ""
	@echo "============================================"
	@echo "Phase 1: vLLM 벤치마크"
	@echo "============================================"
	@echo "🐳 vLLM 컨테이너 시작 중..."
	@cd docker_composes && docker-compose -f docker-compose.vllm.yml up -d
	@echo "⏳ vLLM 서버 준비 대기 중..."
	@timeout=300; \
	elapsed=0; \
	while [ $$elapsed -lt $$timeout ]; do \
		if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then \
			echo "✅ vLLM 서버 준비 완료 ($$elapsed초)"; \
			break; \
		fi; \
		sleep 5; \
		elapsed=$$((elapsed + 5)); \
		echo "  대기 중... ($$elapsed/$$timeout초)"; \
	done; \
	if [ $$elapsed -ge $$timeout ]; then \
		echo "❌ vLLM 서버 시작 타임아웃 ($$timeout초 초과)"; \
		cd docker_composes && docker-compose -f docker-compose.vllm.yml logs --tail=50; \
		cd docker_composes && docker-compose -f docker-compose.vllm.yml down; \
		exit 1; \
	fi
	@echo "📈 vLLM 벤치마크 실행 중 (포트 8000)..."
	@BENCHMARK_PORT=8000 BENCHMARK_OUTPUT=results_vllm.json python3 benchmark_llm.py || true
	@if [ -f results_vllm.json ]; then \
		echo "✅ vLLM 벤치마크 완료 (results_vllm.json)"; \
	else \
		echo "❌ vLLM 벤치마크 실패"; \
	fi
	@echo "🛑 vLLM 컨테이너 종료 중..."
	@cd docker_composes && docker-compose -f docker-compose.vllm.yml down
	@echo ""
	@echo "============================================"
	@echo "Phase 2: TensorRT-LLM 벤치마크"
	@echo "============================================"
	@echo "🐳 TensorRT-LLM 컨테이너 시작 중..."
	@cd docker_composes && docker-compose -f docker-compose.tensorrt-llm.yml up -d
	@echo "⏳ TensorRT-LLM 서버 준비 대기 중..."
	@timeout=600; \
	elapsed=0; \
	while [ $$elapsed -lt $$timeout ]; do \
		if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then \
			echo "✅ TensorRT-LLM 서버 준비 완료 ($$elapsed초)"; \
			break; \
		fi; \
		sleep 5; \
		elapsed=$$((elapsed + 5)); \
		echo "  대기 중... ($$elapsed/$$timeout초)"; \
	done; \
	if [ $$elapsed -ge $$timeout ]; then \
		echo "❌ TensorRT-LLM 서버 시작 타임아웃 ($$timeout초 초과)"; \
		cd docker_composes && docker-compose -f docker-compose.tensorrt-llm.yml logs --tail=50; \
		cd docker_composes && docker-compose -f docker-compose.tensorrt-llm.yml down; \
		exit 1; \
	fi
	@echo "📈 TensorRT-LLM 벤치마크 실행 중 (포트 8001)..."
	@BENCHMARK_PORT=8001 BENCHMARK_OUTPUT=results_tensorrt.json python3 benchmark_llm.py || true
	@if [ -f results_tensorrt.json ]; then \
		echo "✅ TensorRT-LLM 벤치마크 완료 (results_tensorrt.json)"; \
	else \
		echo "❌ TensorRT-LLM 벤치마크 실패"; \
	fi
	@echo "🛑 TensorRT-LLM 컨테이너 종료 중..."
	@cd docker_composes && docker-compose -f docker-compose.tensorrt-llm.yml down
	@echo ""
	@echo "============================================"
	@echo "✅ 자동 벤치마크 완료!"
	@echo "============================================"
	@echo "📊 결과 파일:"
	@ls -lh results_*.json 2>/dev/null || echo "  결과 파일이 없습니다"
	@echo ""

# 벤치마크 결과 비교
.PHONY: compare
compare:
	@echo "📊 결과 비교 중..."
	@if [ -f results_vllm.json ] && [ -f results_tensorrt.json ]; then \
		echo ""; \
		echo "=== vLLM 결과 ==="; \
		cat results_vllm.json | python3 -m json.tool | grep -A 10 "throughput_tokens_per_sec"; \
		echo ""; \
		echo "=== TensorRT-LLM 결과 ==="; \
		cat results_tensorrt.json | python3 -m json.tool | grep -A 10 "throughput_tokens_per_sec"; \
	else \
		echo "❌ 결과 파일이 없습니다. 먼저 'make benchmark'를 실행하세요."; \
	fi

# ============================================
# 모델 준비 안내
# ============================================
# 모델은 HuggingFace에서 수동으로 다운로드해야 합니다:
#
# 방법 1: HuggingFace CLI
#   huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/models/Qwen3-8B
#
# 방법 2: Python
#   from transformers import AutoModelForCausalLM, AutoTokenizer
#   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
#   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
#   model.save_pretrained("/path/to/models/Qwen3-8B")
#   tokenizer.save_pretrained("/path/to/models/Qwen3-8B")
