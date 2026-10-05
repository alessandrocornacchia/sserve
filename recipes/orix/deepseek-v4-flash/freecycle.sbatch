#!/bin/bash --login
# sserve: DeepSeek-V4-Flash, 4 x H200 on freecycle-h200 (preemptible, requeued), 128k ctx
#
# Runs on any idle H200 (higher group limits than batch), but any non-freecycle job
# can preempt it: SIGTERM, 60 s grace, then the job is requeued and serve.sh starts
# again (same job id, port and API key; likely a different node). After a requeue,
# run `sserve up deepseek-v4-flash:freecycle` again to re-attach the tunnel.
# Same as `sbatch-free`. Low-utilisation auto-cancel threshold is 75%.
#
#SBATCH --partition=freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:4

# Under sserve the recipe dir is exported; with a plain sbatch, fall back to the deploy dir.
exec bash "${SSERVE_RECIPE_DIR:-$HOME/.vllm/deploy/deepseek-v4-flash}/serve.sh"
