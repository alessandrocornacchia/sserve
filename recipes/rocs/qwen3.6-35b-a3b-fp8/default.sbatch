#!/bin/bash --login
# sserve: Qwen3.6-35B-A3B-FP8 on 1 x A100 80GB (normal QoS, not preemptible), tool calling + reasoning
#
# ROCS policy (https://sands.kaust.edu.sa/internal/rocs-testbed/slurm-environment/):
# - normal QoS caps each user at 2 x A100 (8 x V100, 8 x P100) at once.
# - A GPU under 15% utilisation for two consecutive 1 h windows gets the job
#   cancelled: `sserve down` when done.
# gpu_a100_80gb matches the SXM node (mcnode22) and the PCIe one (acclnode06);
# PCIe is ~10% slower on a single card, which does not matter here.
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb

# Under sserve the recipe dir is exported; with a plain sbatch, fall back to the deploy dir.
exec bash --login "${SSERVE_RECIPE_DIR:-$HOME/.vllm/deploy/qwen3.6-35b-a3b-fp8}/serve.sh"
