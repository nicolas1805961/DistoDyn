#!/bin/bash
 
#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8    # jackhmmer default requirement
#SBATCH --mem=50G
#SBATCH --array=1-1
#SBATCH --output=/dev/null  # Prevent SLURM from making its own output file
##SBATCH --qos=gpu
 
#---- Job Name
#SBATCH -J 1STU_af2_job

#export TF_FORCE_UNIFIED_MEMORY=1
#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
#export OPENMM_PLATFORM=CPU

# Get the filename from list_dir (one filename per line)
FILE_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" list_dir_sp)
BASENAME=$(basename "${FILE_NAME}" .fasta)
 
#INPUT_FASTA=/pasteur/appa/scratch/public/edeveaud/multimer.fasta
BASE_DIR=~/alphafold2/data/sequences/fasta_sequences/
#OUTPUT_DIR=/pasteur/appa/scratch/public/edeveaud/multimer_out
OUTPUT_DIR=/pasteur/appa/scratch/nportal/af2_sp_results
PRESET_MODEL=monomer # multimer model
PRESET_DB=full_dbs # <reduced_dbs|full_dbs>: Choose preset MSA database configuration

module load alphafold/2.3.2

# Redirect all output manually
mkdir -p slurm_OUT
exec > "slurm_OUT/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${BASENAME}.out" 2>&1

INPUT_PATH="${BASE_DIR}/${FILE_NAME}"

 
/pasteur/appa/homes/nportal/alphafold2/alphafold --fasta_paths ${INPUT_PATH} \
        --max_template_date 2025-07-10 \
        --output_dir ${OUTPUT_DIR} \
        --model_preset monomer --data_dir ${ALPHAFOLD_DATA} \
        --uniref90_database_path ${ALPHAFOLD_DATA}/uniref90/uniref90.fasta \
        --mgnify_database_path ${ALPHAFOLD_DATA}/mgnify/mgy_clusters.fa \
        --pdb70_database_path ${ALPHAFOLD_DATA}/pdb70/pdb70 \
        --template_mmcif_dir ${ALPHAFOLD_DATA}/pdb_mmcif/mmcif_files \
        --obsolete_pdbs_path ${ALPHAFOLD_DATA}/pdb_mmcif/obsolete.dat \
        --bfd_database_path ${ALPHAFOLD_DATA}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
        --uniref30_database_path /pasteur/appa/scratch/ssen/Uniref30_2023_02/UniRef30_2023_02 \
        --db_preset full_dbs \
        --use_gpu_relax=true \
        --verbosity 1
