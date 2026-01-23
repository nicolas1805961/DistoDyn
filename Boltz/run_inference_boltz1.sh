#!/bin/bash

#SBATCH -N 1
#SBATCH --partition=baies
#SBATCH --account=baies
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1,gmem:50G
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=1-456
#SBATCH -J boltz
#SBATCH --output=/dev/null  # Prevent SLURM from making its own output file

BASE_DIR=~/boltz/data/Sequences/Boltz_input_paper
OUTPUT_DIR=/pasteur/appa/scratch/nportal/results_boltz1

# Get the filename from list_dir (one filename per line)
FILE_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" list_dir)
BASENAME=$(basename "${FILE_NAME}" .fasta)

# Redirect all output manually
mkdir -p slurm_OUT
exec > "slurm_OUT/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${BASENAME}.out" 2>&1

INPUT_PATH="${BASE_DIR}/${FILE_NAME}"
boltz predict "${INPUT_PATH}" --use_msa_server --model boltz2 --out_dir "${OUTPUT_DIR}"
