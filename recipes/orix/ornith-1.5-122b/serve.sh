#!/bin/bash
# Shared vLLM launch for ornith-1.5-122b recipes (qwen3_5_moe, BF16 ~250 GB, TP=4).
# Flags follow the model card (qwen3_xml tool parser, qwen3 reasoning parser) minus
# --trust-remote-code (the repo ships no custom code). Revision is pinned.
set -eo pipefail

# --- sserve contract: listen on $VLLM_PORT, require the per-run API key -----
: "${VLLM_PORT:=8000}"
[[ -n "${SSERVE_KEY_FILE:-}" ]] && export VLLM_API_KEY="$(<"$SSERVE_KEY_FILE")"
# ----------------------------------------------------------------------------

source "$HOME/venvs/vllm/bin/activate"
export HF_HUB_OFFLINE=1
# Node driver is CUDA 12.8; vLLM 0.24.0 ships cu129 wheels -> forward-compat libcuda.
export LD_LIBRARY_PATH=/mnt/data/u/cornaca/cuda-compat-12-9/root/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
export VLLM_ENGINE_READY_TIMEOUT_S=3600

echo "node: $(hostname)  port: $VLLM_PORT  gpus: $CUDA_VISIBLE_DEVICES  HF_HOME: ${HF_HOME:-}" \
     " partition: ${SLURM_JOB_PARTITION:-}  restart: ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*"
python -c "import vllm;print('vllm', vllm.__version__)"

exec vllm serve ornith-ai/Ornith-1.5-122B \
    --revision 7681ef608624e7532d8164128bcffdbdbaf1b1af \
    --served-model-name ornith-ai/Ornith-1.5-122B \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
    --max-model-len "${MAX_MODEL_LEN:-262144}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    ${VLLM_EXTRA_ARGS:-}
