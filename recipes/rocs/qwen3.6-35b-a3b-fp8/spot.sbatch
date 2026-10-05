#!/bin/bash --login
# sserve: Qwen3.6-35B-A3B-FP8 on 1 x A100 80GB, spot QoS (preemptible, requeued; uses idle reserved GPUs)
#
# spot has no per-user GPU cap, and --signal=R:... lets the job run on idle
# reserved GPUs too. Any normal-QoS job or reservation holder can preempt it;
# SLURM then requeues it (ROCS default): same job id, port and key, maybe
# another node. Re-run `sserve up qwen3.6-35b-a3b-fp8:spot -d` to re-attach.
#
#SBATCH --qos=spot
#SBATCH --signal=R:TERM@60
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb

# Under sserve the recipe dir is exported; with a plain sbatch, fall back to the deploy dir.
exec bash --login "${SSERVE_RECIPE_DIR:-$HOME/.vllm/deploy/qwen3.6-35b-a3b-fp8}/serve.sh"
