#!/bin/bash
#SBATCH -N 1
#SBATCH --account=baies
#SBATCH --partition=baies
#SBATCH --cpus-per-task=1
#SBATCH -J submit_chunks
#SBATCH --output=submit_chunks_%j.out

CHUNK_SIZE=10000      # Number of files per chunk
MAX_QUEUED=5          # Max jobs allowed in queue for your user
#BASE_DIR="/pasteur/appa/scratch/nportal/boltz/stability_prediction/temp"
BASE_DIR="/pasteur/appa/scratch/nportal/boltz/stability_prediction/distances"

mkdir -p sp_list

echo "Creating chunk lists from $BASE_DIR..."

find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type f -name "protein_*_*.pkl" -printf "%f\n" \
    | sed 's/\.pkl$//' \
    | sort \
    | split -l "$CHUNK_SIZE" -d - sp_list/sp_list_mutated_

#find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type f | sort \
#    | split -l "$CHUNK_SIZE" -d - sp_list/sp_list_mutated_

chunk_files=(sp_list/sp_list_mutated_*)
total_chunks=${#chunk_files[@]}
count=0

for LIST in "${chunk_files[@]}"; do
    NUM_LINES=$(wc -l < "$LIST")

    # Wait if too many jobs are queued before submitting next chunk
    while [ $(squeue -u $USER | grep -c 'boltz') -ge $MAX_QUEUED ]; do
        echo "$(date) - Too many jobs queued ($MAX_QUEUED). Waiting 60s..."
        sleep 60
    done

    echo "Submitting $LIST with $NUM_LINES jobs..."
    job_output=$(sbatch --exclude=maestro-3478,maestro-3485,maestro-3493 --array=1-${NUM_LINES} --export=LIST_FILE=$LIST child_slurm_job.sh)
    echo "$job_output"

    ((count++))
    percent=$(( 100 * count / total_chunks ))
    progress_width=50
    filled=$(( percent * progress_width / 100 ))
    empty=$(( progress_width - filled ))
    bar=$(printf "%0.s#" $(seq 1 $filled))
    space=$(printf "%0.s-" $(seq 1 $empty))
    printf "\rProgress: [%s%s] %d%% (%d/%d chunks)" "$bar" "$space" "$percent" "$count" "$total_chunks"
done

echo -e "\nAll chunks submitted."
