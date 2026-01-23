#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
##SBATCH --mem=50G
#SBATCH --array=1-5318
#SBATCH -J misato
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load Python/3.11.5

PDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_list) #list_dir

mkdir -p slurm_OUT_distance_distogram_new_new_new
exec > "slurm_OUT_distance_distogram_new_new_new/${PDB_ID}.out" 2>&1

python3 -u create_pt_distance_distogram_new_new_new.py --threshold 0.0001 -pdb "${PDB_ID}"
