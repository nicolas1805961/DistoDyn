#!/bin/bash

#SBATCH -N 1
##SBATCH --partition=baies
##SBATCH --account=baies
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,gmem:50G
##SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=1-40
#SBATCH -J boltz
#SBATCH --output=/dev/null  # Prevent SLURM from making its own output file

BASE_DIR=/pasteur/appa/homes/nportal/misato-dataset/boltz_inputs_yaml
OUTPUT_DIR=/pasteur/appa/scratch/nportal/MISATO/Binding_site/inference_boltz1

# Get the filename from list_dir (one filename per line)
FILE_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" missing_files) #list_dir
BASENAME=$(basename "${FILE_NAME}" .yaml)

# Redirect all output manually
mkdir -p slurm_OUT_boltz1_error_2
exec > "slurm_OUT_boltz1_error_2/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${BASENAME}.out" 2>&1

INPUT_PATH="${BASE_DIR}/${FILE_NAME}"

module load Python/3.11.5
#module load cuda/12.8

#export LD_LIBRARY_PATH=/opt/gensoft/exe/cuda/12.9/lib64:$HOME/.local/lib/python3.11/site-packages/cuequivariance_ops/lib:$LD_LIBRARY_PATH

boltz predict "${INPUT_PATH}" --use_msa_server --model boltz1 --out_dir "${OUTPUT_DIR}"