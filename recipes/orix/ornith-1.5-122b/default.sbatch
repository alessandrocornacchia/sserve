#!/bin/bash --login
# sserve: Ornith-1.5-122B BF16 @7681ef60, 4 x H200 TP=4 on batch-h200 (not preemptible), 256k ctx; 30 h benchmark serving
#
# Batch account had 4 GPU / 96 CPU / 1182 GB available at submit (2026-10-21 20:00 UTC).
# ORIX auto-cancels GPUs < 75% util (1 h avg) for 2 consecutive hours: keep it busy or sserve down.
#
#SBATCH --partition=batch-h200
#SBATCH --qos=batch
#SBATCH --time=1-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:nvidia_h200:4

exec bash "${SSERVE_RECIPE_DIR:-$HOME/.vllm/deploy/ornith-1.5-122b}/serve.sh"
