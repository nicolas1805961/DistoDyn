#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
##SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1,gmem:50G
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --array=1-16972
#SBATCH -J misato
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Get the filename from list_dir (one filename per line)
PDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" protein_list) #list_dir

# Prepare output folder and redirect stdout/stderr
mkdir -p slurm_OUT_distances
exec > "slurm_OUT_distances/${PDB_ID}.out" 2>&1

module load Python/3.11.5
python3 get_distance.py "${PDB_ID}"
