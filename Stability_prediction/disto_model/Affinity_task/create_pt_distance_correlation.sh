#!/bin/bash

#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
##SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1,gmem:50G
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --array=1-1
#SBATCH -J misato
#SBATCH --output=logs/create_pt_distance_correlation_log.out
#SBATCH --error=logs/create_pt_distance_correlation_log.out

module load Python/3.11.5
python3 -u create_pt_distance_correlation.py
