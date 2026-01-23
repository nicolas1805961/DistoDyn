#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
##SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1,gmem:50G
#SBATCH --gres=gpu:1
##SBATCH --mem=50G
#SBATCH --array=1-16972
#SBATCH -J misato
#SBATCH --output=logs/get_distance_boltz.out
#SBATCH --error=logs/get_distance_boltz.out

# Get the filename from list_dir (one filename per line)
PDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_ids.txt) #list_dir

module load Python/3.11.5
python3 get_distance_boltz.py "${PDB_ID}"
