#!/bin/bash --login
# sserve: Qwen3-0.6B BF16 on 1 x H100 (batch-h100), 32k ctx; short smoke tests only
#
# Under sserved the job script is generated from the operator's profile and this
# file is not used; it keeps the label deployable with plain `sserve up`.
#
#SBATCH --partition=batch-h100
#SBATCH --qos=batch
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1

exec bash "${SSERVE_RECIPE_DIR:-$HOME/.vllm/deploy/qwen3-0.6b}/serve.sh"
