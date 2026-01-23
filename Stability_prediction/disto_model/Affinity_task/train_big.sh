#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,gmem:50G
#SBATCH --mem=50G
#SBATCH --array=1-1
#SBATCH -J boltz
#SBATCH --output=slurm-%A_%a.out

module load Python/3.11.5
python3 train_misato_affinity.py config${SLURM_ARRAY_TASK_ID}.yaml