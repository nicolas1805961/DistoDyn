#!/bin/bash

#!/bin/bash
#SBATCH --job-name=colabfold_search
#SBATCH --partition=common
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
##SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

module load Python/3.11.5

# Add gdl folder to PYTHONPATH
export PYTHONPATH=/pasteur/appa/homes/nportal/misato-dataset/understanding-oversquashing:$PYTHONPATH

# Set max steps for SDRF (adjust as needed)
MAX_STEPS=5000

# Set paths to folders
PT_FOLDER="/pasteur/appa/scratch/nportal/MISATO/Binding_site/pt_folder_distances_2/test"
GT_FOLDER="/pasteur/appa/scratch/nportal/MISATO/Binding_site/binding_sites/test"
TMP_ROOT="/pasteur/appa/scratch/nportal/MISATO/Binding_site/pt_folder_distances_rewired_${MAX_STEPS}_2/test"

# Create the folder if it doesn’t exist
mkdir -p "$TMP_ROOT"

# Run the Python script with all arguments
python3 -u ICML_review_pool.py "$PT_FOLDER" "$GT_FOLDER" "$TMP_ROOT" --max_steps "$MAX_STEPS" --num_workers 64