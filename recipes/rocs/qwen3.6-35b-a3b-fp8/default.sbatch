#!/bin/bash --login
# sserve: Qwen3.6-35B-A3B-FP8 on 1 x A100 80GB, tool calling + reasoning
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb

set -eo pipefail

# --- sserve contract: listen on $VLLM_PORT, require the per-run API key -----
: "${VLLM_PORT:=8000}"
[[ -n "${SSERVE_KEY_FILE:-}" ]] && export VLLM_API_KEY="$(<"$SSERVE_KEY_FILE")"
# ----------------------------------------------------------------------------

mamba activate vllm-0.19-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

exec vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data
