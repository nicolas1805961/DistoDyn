#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
##SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1,gmem:50G
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --array=1-145
#SBATCH -J misato
#SBATCH --output=/dev/null  # Prevent SLURM from making its own output file

BASE_DIR=/pasteur/appa/homes/nportal/misato-dataset/boltz_inputs_yaml_2
OUTPUT_DIR=/pasteur/appa/scratch/nportal/MISATO/inference_2

# Get the filename from list_dir (one filename per line)
FILE_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" yaml_list_2) #list_dir
BASENAME=$(basename "${FILE_NAME}" .yaml)

# Redirect all output manually
mkdir -p slurm_OUT_2
exec > "slurm_OUT_2/${BASENAME}.out" 2>&1

INPUT_PATH="${BASE_DIR}/${FILE_NAME}"
#boltz predict "${INPUT_PATH}" --model boltz2 --out_dir "${OUTPUT_DIR}"
boltz predict "${INPUT_PATH}" --use_msa_server --model boltz2 --out_dir "${OUTPUT_DIR}"
