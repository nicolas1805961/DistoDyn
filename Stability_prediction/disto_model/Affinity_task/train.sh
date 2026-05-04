#!/bin/bash
#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
##SBATCH --mem=50G
#SBATCH --array=1-10
#SBATCH -J boltz
#SBATCH --output=slurm-%A_%a.out

module load Python/3.11.5
python3 train_misato_affinity.py config${SLURM_ARRAY_TASK_ID}.yaml