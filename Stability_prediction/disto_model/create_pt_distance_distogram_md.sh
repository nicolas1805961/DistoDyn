#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
##SBATCH --mem=50G
#SBATCH --array=1-16972
#SBATCH -J misato
#SBATCH --output=logs/create_pt_distance_distogram_md_log.out
#SBATCH --error=logs/create_pt_distance_distogram_md_log.out

module load Python/3.11.5

PDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_ids.txt) #list_dir

python3 -u create_pt_distance_distogram_md.py --threshold 0.0001 -pdb "${PDB_ID}"
