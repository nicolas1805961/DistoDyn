#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
##SBATCH --mem=50G
#SBATCH --array=1-16972
#SBATCH -J misato
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load Python/3.11.5

PDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_ids.txt) #list_dir

mkdir -p slurm_OUT_distance_distance_random
exec > "slurm_OUT_distance_distance_random/${PDB_ID}.out" 2>&1

python3 -u create_pt_distance_distance_random.py --threshold 0.0001 -pdb "${PDB_ID}"
