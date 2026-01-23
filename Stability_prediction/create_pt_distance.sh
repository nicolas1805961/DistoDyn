#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
##SBATCH --mem=50G
#SBATCH --array=1-16972
#SBATCH -J misato
#SBATCH --output=log/create_distance_%A_%a.out
#SBATCH --error=log/create_distance_%A_%a.out

module load Python/3.11.5

PDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_ids.txt)

python3 -u create_pt_distance.py --pdb "${PDB_ID}"