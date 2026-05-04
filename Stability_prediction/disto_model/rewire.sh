#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
##SBATCH --mem=50G
#SBATCH --array=1-1589 #1610, 1589, 13750
#SBATCH -J misato
#SBATCH --output=logs/rewire_%A_%a.out
#SBATCH --error=logs/rewire_%A_%a.out

module load Python/3.11.5

# Add gdl folder to PYTHONPATH
export PYTHONPATH=/pasteur/appa/homes/nportal/misato-dataset/understanding-oversquashing:$PYTHONPATH

# Get the protein name from the pdb_ids.txt file
PDB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_ids_val.txt)

# Set max steps for SDRF (adjust as needed)
MAX_STEPS=10000

# Set paths to folders
PT_FOLDER="/pasteur/appa/scratch/nportal/MISATO/Binding_site/pt_folder_distances_2/val"
GT_FOLDER="/pasteur/appa/scratch/nportal/MISATO/Binding_site/binding_sites/val"
TMP_ROOT="/pasteur/appa/scratch/nportal/MISATO/Binding_site/pt_folder_distances_rewired_${MAX_STEPS}_2/val"

# Create the folder if it doesn’t exist
mkdir -p "$TMP_ROOT"

# Run the Python script with all arguments
python3 -u ICML_review.py "$PDB_ID" "$PT_FOLDER" "$GT_FOLDER" "$TMP_ROOT" --max_steps "$MAX_STEPS"