#!/bin/bash --login
# sserve: DeepSeek-V4-Flash, 4 x H200 on the pi-caninim reservation (guaranteed, not preemptible)
#
# pi-caninim is a half-node reservation (orix-worker-h200-7, QoS cap 4 GPU / 64 CPU /
# 719 GB for the whole group), so this takes all of it: group-mates' jobs on it
# make this one pend, and vice versa. Same as `sbatch-pi`.
# Low-utilisation auto-cancel threshold on PI partitions is 50% (2 x 1 h windows).
#
#SBATCH --partition=pi-caninim
#SBATCH --qos=pi-caninim
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:4

# Under sserve the recipe dir is exported; with a plain sbatch, fall back to the deploy dir.
exec bash "${SSERVE_RECIPE_DIR:-$HOME/.vllm/deploy/deepseek-v4-flash}/serve.sh"
