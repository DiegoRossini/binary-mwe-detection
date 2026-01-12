#!/bin/bash
#SBATCH --job-name=mwe
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs outputs

if [ -n "$SLURM_JOB_ID" ]; then
    exec > "logs/mwe_${SLURM_JOB_ID}.out" 2> "logs/mwe_${SLURM_JOB_ID}.err"
fi

echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

python main.py

echo "Done: $(date)"
