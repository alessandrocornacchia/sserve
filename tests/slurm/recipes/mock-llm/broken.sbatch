#!/bin/bash
# sserve: exits before serving, to test how sserve reports a failed job
#SBATCH --time=5:00
#SBATCH --ntasks=1

echo "mock-llm:broken: failing on purpose" >&2
exit 1
