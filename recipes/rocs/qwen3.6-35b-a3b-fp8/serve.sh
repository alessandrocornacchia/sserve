#!/bin/bash --login
# Shared vLLM launch for every qwen3.6-35b-a3b-fp8 recipe on ROCS. The *.sbatch files
# only differ in where the job runs (GPU type, QoS); they exec this script with
# `bash --login`, which mamba needs to activate the environment.

set -eo pipefail

# --- sserve contract: listen on $VLLM_PORT, require the per-run API key -----
: "${VLLM_PORT:=8000}"
[[ -n "${SSERVE_KEY_FILE:-}" ]] && export VLLM_API_KEY="$(<"$SSERVE_KEY_FILE")"
# ----------------------------------------------------------------------------

mamba activate vllm-0.19-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# restart > 0 = requeued after a preemption (spot QoS)
echo "node: $(hostname)  port: $VLLM_PORT  gpus: $CUDA_VISIBLE_DEVICES  qos: ${SLURM_JOB_QOS:-}" \
     " restart: ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --tensor-parallel-size "${TP_SIZE:-1}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data
