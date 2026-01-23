#!/bin/bash
#SBATCH -N 1
#SBATCH -A mxf@a100
#SBATCH -C a100
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1                    # we request one node
#SBATCH --hint=nomultithread  
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1
#SBATCH --array=1-15
#SBATCH -J boltz
#SBATCH --time=20:00:00          # 48:00:00 temps maximum d'execution demande (HH:MM:SS) 00:05:00 20:00:00 
#SBATCH --output=slurm-%A_%a.out  # Prevent SLURM from making its own output file

module load pytorch-gpu/py3/2.8.0
python3 train_megascale.py config${SLURM_ARRAY_TASK_ID}.yaml