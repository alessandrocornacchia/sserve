#!/bin/bash --login
# sserve: DeepSeek-V4-Flash, 4 x H200 on batch-h200 (shared pool, not preemptible), 128k ctx
#
# ORIX policy (https://orix.kaust.edu.sa/cluster_policy/):
# - batch QoS caps the whole PI account at 8 GPU / 128 CPU / 1438 GB at once; if a
#   group-mate holds that, the job pends with MaxMemoryPerAccount / MaxGRESPerAccount.
#   Check with /opt/orix/utils/susage, or use :pi / :freecycle.
# - GPUs averaging < 75% utilisation for two consecutive 1 h windows get the job
#   cancelled (and repeated idle reservations risk suspension): `sserve down` when done.
#
#SBATCH --partition=batch-h200
#SBATCH --qos=batch
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:4

# Under sserve the recipe dir is exported; with a plain sbatch, fall back to the deploy dir.
exec bash "${SSERVE_RECIPE_DIR:-$HOME/.vllm/deploy/deepseek-v4-flash}/serve.sh"
