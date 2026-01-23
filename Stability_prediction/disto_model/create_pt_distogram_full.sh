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

PDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_ids.txt)

THRESHOLD=0.0001

# Prepare output folder named according to the threshold
OUT_DIR="slurm_OUT_distogram_full_${THRESHOLD}"
mkdir -p "$OUT_DIR"

# Redirect stdout/stderr
exec > "${OUT_DIR}/${PDB_ID}.out" 2>&1

echo "Running job $SLURM_ARRAY_TASK_ID with threshold $THRESHOLD"

python3 -u create_pt_distogram.py --threshold "$THRESHOLD" --pdb "${PDB_ID}"