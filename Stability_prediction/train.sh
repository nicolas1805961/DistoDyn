#!/bin/bash
#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=1-1
#SBATCH -J boltz
#SBATCH --output=slurm-%A_%a.out

module load Python/3.11.5
python3 train_megascale.py config${SLURM_ARRAY_TASK_ID}.yaml