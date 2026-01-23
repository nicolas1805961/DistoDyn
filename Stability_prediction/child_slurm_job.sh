#!/bin/bash
#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -J boltz
#SBATCH --output=log_child_job  # Prevent SLURM from making its own output file
#SBATCH --error=log_child_job  # Prevent SLURM from making its own output file

# Get the filename from the chunk list
FILE_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")
BASENAME=$(basename "${FILE_NAME}")

# Redirect all output manually
#mkdir -p slurm_OUT_distance
#exec > "slurm_OUT_distance/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${BASENAME}.out" 2>&1

INPUT_PATH="${FILE_NAME}"

module load Python/3.11.5
python3 -u create_pt_distance_distogram.py --threshold 0.0001 -pdb "${INPUT_PATH}"
