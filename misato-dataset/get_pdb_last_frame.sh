#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
##SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1,gmem:50G
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --array=1-16972
#SBATCH -J misato
#SBATCH --output=logs/get_pdb_last_frame.log
#SBATCH --error=logs/get_pdb_last_frame.log

# Get the filename from list_dir (one filename per line)
FILE_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pdb_ids.txt) #list_dir

module load Python/3.11.5
#python3 pdb_to_sequnce.py -pdb 1FO0
python3 get_pdb_last_frame.py -pdb "${FILE_NAME}"
