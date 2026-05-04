#!/bin/bash
 
#SBATCH -N 1
##SBATCH --account=baies
##SBATCH --partition=baies
#SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --gres=gpu:1,gmem:50G
#SBATCH --cpus-per-task=8    # jackhmmer default requirement
##SBATCH --mem=50G
#SBATCH --array=1-9
#SBATCH --output=/dev/null  # Prevent SLURM from making its own output file
##SBATCH --qos=gpu
 
#---- Job Name
#SBATCH -J 1STU_af2_job

#export TF_FORCE_UNIFIED_MEMORY=1
#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
#export OPENMM_PLATFORM=CPU

# Get the filename from list_dir (one filename per line)
FILE_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" missing_files)
BASENAME=$(basename "${FILE_NAME}" .fasta)
 
#INPUT_FASTA=/pasteur/appa/scratch/public/edeveaud/multimer.fasta
#BASE_DIR=/pasteur/appa/homes/nportal/misato-dataset/boltz_inputs_fasta/
BASE_DIR=/pasteur/appa/scratch/nportal/af2/fasta_sequences
#OUTPUT_DIR=/pasteur/appa/scratch/public/edeveaud/multimer_out
OUTPUT_DIR=/pasteur/appa/scratch/nportal/af2/vincent_paper
PRESET_MODEL=multimer # multimer model
PRESET_DB=full_dbs # <reduced_dbs|full_dbs>: Choose preset MSA database configuration

module load alphafold/2.3.2

# Redirect all output manually
mkdir -p slurm_OUT_vincent
exec > "slurm_OUT_vincent/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${BASENAME}.out" 2>&1

INPUT_PATH="${BASE_DIR}/${FILE_NAME}"

 
#/pasteur/appa/homes/nportal/alphafold2/alphafold --fasta_paths ${INPUT_PATH} \
alphafold --fasta_paths ${INPUT_PATH} \
        --max_template_date 2026-01-01 \
        --output_dir ${OUTPUT_DIR} \
        --model_preset multimer --data_dir ${ALPHAFOLD_DATA} \
        --uniref90_database_path ${ALPHAFOLD_DATA}/uniref90/uniref90.fasta \
        --mgnify_database_path ${ALPHAFOLD_DATA}/mgnify/mgy_clusters.fa \
        --template_mmcif_dir ${ALPHAFOLD_DATA}/pdb_mmcif/mmcif_files \
        --obsolete_pdbs_path ${ALPHAFOLD_DATA}/pdb_mmcif/obsolete.dat \
        --bfd_database_path ${ALPHAFOLD_DATA}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
        --uniref30_database_path /pasteur/appa/scratch/ssen/Uniref30_2023_02/UniRef30_2023_02 \
        --db_preset full_dbs --uniprot_database_path ${ALPHAFOLD_DATA}/uniprot/uniprot.fa \
        --pdb_seqres_database_path ${ALPHAFOLD_DATA}/pdb_seqres/pdb_seqres.txt \
        --use_gpu_relax=true \
        --verbosity 1
