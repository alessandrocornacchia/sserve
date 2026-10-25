#!/bin/bash
# sserve: fake vLLM (echo server) for the Docker test cluster
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

: "${VLLM_PORT:=8000}"
[[ -n "${SSERVE_KEY_FILE:-}" ]] && export VLLM_API_KEY="$(<"$SSERVE_KEY_FILE")"

exec python3 "${SSERVE_RECIPE_DIR:-/data/sserve/deploy/mock-llm}/serve.py" --port "$VLLM_PORT"
