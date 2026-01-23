#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --array=1-414   # <-- adjust to number of lines in your list file
#SBATCH -J mutate_boltz
#SBATCH --output=/dev/null  # Prevent SLURM from making its own output file

# Path to the list of input folders
LIST_FILE=list_input_paths   # one folder path per line

# Get the correct folder and fasta path for this array task
INPUT_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")

# Derive a nice name for the log file
BASENAME=$(basename "${INPUT_PATH}")

# Redirect output manually
mkdir -p slurm_OUT
exec > "slurm_OUT/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${BASENAME}.out" 2>&1

# Load Python
module load Python/3.11.5

# Run your script
python3 mutate_boltz.py \
    --input_path "${INPUT_PATH}"
