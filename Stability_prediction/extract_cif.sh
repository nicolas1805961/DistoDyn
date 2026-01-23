#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
##SBATCH --partition=gpu
##SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1,gmem:50G
##SBATCH --gres=gpu:1
##SBATCH --mem=50G
#SBATCH --array=1-16972
#SBATCH -J misato
#SBATCH --output=extract_cif.log
#SBATCH --error=extract_cif.log

# Get the filename from list_dir (one filename per line)
CIF_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subfolder.txt) #list_dir

module load Python/3.11.5
python3 extract_cif_files.py "${CIF_PATH}" /pasteur/appa/scratch/nportal/boltz/pandora_cif
#python3 get_distance.py "${PDB_ID}"
