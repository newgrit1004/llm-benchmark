#!/bin/bash
set -e

echo "============================================"
echo "🚀 TensorRT-LLM 자동 초기화 시작"
echo "============================================"
echo ""

# Configuration from environment variables (with defaults)
MODEL_DIR="${MODEL_DIR:-/models/Qwen3-8B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/engines/qwen3-8b/checkpoint}"
ENGINE_DIR="${ENGINE_DIR:-/engines/qwen3-8b/engine}"
PORT="${TENSORRT_PORT:-8001}"
DTYPE="${DTYPE:-bfloat16}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-256}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-4096}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"
TP_SIZE="${TENSORRT_TP_SIZE:-1}"
GEMM_PLUGIN="${TENSORRT_GEMM_PLUGIN:-bfloat16}"

# Step 1: Checkpoint Conversion
if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
    echo "✅ 체크포인트가 이미 존재합니다: $CHECKPOINT_DIR"
else
    echo "📝 Step 1: 체크포인트 변환 중..."
    echo "   Model: $MODEL_DIR"
    echo "   Output: $CHECKPOINT_DIR"

    python3 /app/tensorrt_llm/examples/qwen/convert_checkpoint.py \
        --model_dir "$MODEL_DIR" \
        --output_dir "$CHECKPOINT_DIR" \
        --dtype "$DTYPE" \
        --tp_size "$TP_SIZE"

    echo "✅ 체크포인트 변환 완료"
fi

echo ""

# Step 2: TensorRT Engine Build
if [ -d "$ENGINE_DIR" ] && [ "$(ls -A $ENGINE_DIR)" ]; then
    echo "✅ TensorRT 엔진이 이미 존재합니다: $ENGINE_DIR"
else
    echo "🔧 Step 2: TensorRT 엔진 빌드 중..."
    echo "   Checkpoint: $CHECKPOINT_DIR"
    echo "   Output: $ENGINE_DIR"
    echo "   ⚠️  이 작업은 시간이 오래 걸릴 수 있습니다 (5-10분)..."

    trtllm-build \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --gemm_plugin "$GEMM_PLUGIN" \
        --max_batch_size "$MAX_BATCH_SIZE" \
        --max_input_len "$MAX_INPUT_LEN" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --max_num_tokens "$MAX_NUM_TOKENS" \
        --use_cuda_graph

    echo "✅ TensorRT 엔진 빌드 완료"
fi

echo ""

# Step 3: Start TensorRT-LLM Server
echo "🌐 Step 3: TensorRT-LLM 서버 시작 중..."
echo "   Engine: $ENGINE_DIR"
echo "   Tokenizer: $MODEL_DIR"
echo "   Port: $PORT"
echo ""

exec python3 -m tensorrt_llm.commands.serve serve \
    "$ENGINE_DIR" \
    --tokenizer "$MODEL_DIR" \
    --host "${SERVER_HOST:-0.0.0.0}" \
    --port "$PORT" \
    --backend trt \
    --max_batch_size "$MAX_BATCH_SIZE"
