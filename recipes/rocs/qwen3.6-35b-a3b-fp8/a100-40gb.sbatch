#!/bin/bash --login
# sserve: Qwen3.6-35B-A3B-FP8 on 2 x A100 40GB, tensor parallel (normal QoS), tool calling + reasoning
#
# The FP8 weights (~35 GB) do not fit one 40 GB card, so this splits the model
# over two (TP=2), leaving ~20 GB per GPU for the KV cache. The 40 GB nodes
# (mcnode23-25, 12 GPUs) are a bigger pool than the 80 GB ones, but this takes
# both A100s the normal QoS allows per user.
# gpu_sxm: tensor parallel traffic is ~3x slower between PCIe A100s.
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=gpu_a100_40gb&gpu_sxm

export TP_SIZE=2

# Under sserve the recipe dir is exported; with a plain sbatch, fall back to the deploy dir.
exec bash --login "${SSERVE_RECIPE_DIR:-$HOME/.vllm/deploy/qwen3.6-35b-a3b-fp8}/serve.sh"
