#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --array=1-16972
#SBATCH -J misato
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


PDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_ids.txt) #list_dir

echo "Running job $SLURM_ARRAY_TASK_ID"

# Prepare output folder and redirect stdout/stderr
mkdir -p slurm_OUT_distogram_max
exec > "slurm_OUT_distogram_max/${PDB_ID}.out" 2>&1

module load Python/3.11.5
python3 -u create_pt_distogram.py -pdb "${PDB_ID}"