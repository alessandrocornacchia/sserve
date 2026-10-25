#!/bin/bash
# Shared vLLM launch for qwen3-0.6b (Qwen/Qwen3-0.6B, BF16, 1 GPU, 32k ctx): a small
# model for short manager smoke tests. Used by sserved (which generates the job
# script and exports SSERVE_SERVED_MODEL / TENSOR_PARALLEL_SIZE / MAX_MODEL_LEN)
# and by `sserve up qwen3-0.6b` through default.sbatch.
# Tool calls: hermes parser; reasoning: qwen3 parser (per the Qwen3 model card).
# No --trust-remote-code (the model is a stock architecture). Weights are expected
# in $HF_HOME (download once on the login node: `hf download Qwen/Qwen3-0.6B`).
set -eo pipefail

# --- sserve contract: listen on $VLLM_PORT, require the per-run API key -----
: "${VLLM_PORT:=8000}"
[[ -n "${SSERVE_KEY_FILE:-}" ]] && export VLLM_API_KEY="$(<"$SSERVE_KEY_FILE")"
# ----------------------------------------------------------------------------

source "$HOME/venvs/vllm/bin/activate"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Node driver is CUDA 12.8; vLLM 0.24.0 ships cu129 wheels -> forward-compat libcuda.
export LD_LIBRARY_PATH=/mnt/data/u/cornaca/cuda-compat-12-9/root/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH

echo "node: $(hostname)  port: $VLLM_PORT  gpus: ${CUDA_VISIBLE_DEVICES:-}  HF_HOME: ${HF_HOME:-}" \
     " partition: ${SLURM_JOB_PARTITION:-}  restart: ${SLURM_RESTART_COUNT:-0}"

exec vllm serve Qwen/Qwen3-0.6B \
    --served-model-name "${SSERVE_SERVED_MODEL:-Qwen/Qwen3-0.6B}" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3
