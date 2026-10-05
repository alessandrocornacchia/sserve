#!/bin/bash --login
# sserve: DeepSeek-V4-Flash, 1 node x 4 H200, DP=4 + expert parallel, 128k context, MTP spec decoding
#
# Single-node version of the 2 x 8 GPU head/node1 recipe: no head/worker split,
# DP shrinks to 4. vLLM 0.24.0 has no "dspark" speculative method, so this uses
# deepseek_mtp, and max-model-len is capped (default 1M) to fit the KV cache.
#
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --constraint=gpu_h200_sxm

set -eo pipefail

# --- sserve contract: listen on $VLLM_PORT, require the per-run API key -----
: "${VLLM_PORT:=8000}"
[[ -n "${SSERVE_KEY_FILE:-}" ]] && export VLLM_API_KEY="$(<"$SSERVE_KEY_FILE")"
# ----------------------------------------------------------------------------

source "$HOME/venvs/vllm/bin/activate"
export VLLM_USE_RUST_FRONTEND=1
export HF_HUB_OFFLINE=1

echo "node: $(hostname)  port: $VLLM_PORT  gpus: $CUDA_VISIBLE_DEVICES  HF_HOME: ${HF_HOME:-}"

exec vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --data-parallel-size 4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
    --max-model-len 131072 \
    --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}'
