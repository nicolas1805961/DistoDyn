<h1>Full pipeline:</h1>

This script reads all .json in current work directory (can be modified in the snakefile)

module load apptainer\
module load Python/3.11.5\
module load snakemake\
nohup snakemake --executor=slurm --slurm-requeue -j 241 > snakemake.log 2>&1 &\

Outputs are located in alphafold_results/

<h1>Only msa part (modify input/output directories in the script):</h1>

sbatch run_data.sh\

<h1>Only inference (modify input/output directories in the script):</h1>

sbatch run_inference.sh\

<h1>Update install after modifying source code:</h1>

module load apptainer\
./update_install.sh\