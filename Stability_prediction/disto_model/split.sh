#!/bin/bash
#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -J boltz
#SBATCH --output=/dev/null  # Prevent SLURM from making its own output file

BASE_DIR=/pasteur/appa/scratch/nportal/boltz/stability_prediction/results_mutated/

# Get the filename from the chunk list
FILE_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")

# Redirect all output manually
mkdir -p slurm_OUT
exec > "slurm_OUT/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${BASENAME}.out" 2>&1

INPUT_PATH="${BASE_DIR}/${FILE_NAME}"

module load Python/3.11.5
python3 split_real.py "${INPUT_PATH}"