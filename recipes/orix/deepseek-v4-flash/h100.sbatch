#!/bin/bash --login
# sserve: DeepSeek-V4-Flash, 4 x H100 on batch-h100 (shared pool, not preemptible), 128k ctx
#
# Same model and flags as default.sbatch (H200). Per GPU the weights take ~45.5 GiB
# (146.6 GiB routed experts / 4 under EP + 8.8 GiB dense replicated per DP rank),
# leaving ~25 GiB for KV cache + activations on an 80 GB H100 vs ~80 GiB on an H200:
# same max context, fewer concurrent long sequences. Lower MAX_MODEL_LEN if the
# KV cache cannot hold one full-length sequence.
#
# ORIX policy: same batch QoS cap (8 GPU / 128 CPU / 1438 GB for the whole PI
# account, across batch-h100 + batch-h200) and 75% low-utilisation auto-cancel
# as default.sbatch.
#
#SBATCH --partition=batch-h100
#SBATCH --qos=batch
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4

export MAX_MODEL_LEN=131072

# Under sserve the recipe dir is exported; with a plain sbatch, fall back to the deploy dir.
exec bash "${SSERVE_RECIPE_DIR:-$HOME/.vllm/deploy/deepseek-v4-flash}/serve.sh"
