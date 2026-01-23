#!/bin/bash

# Eric Deveaud <edeveaud@pasteur.fr>
# institut Pasteur

#----------------------------------------------------------
#  Wrapper script for alphafold 
#  with default values preset
#----------------------------------------------------------

#---- Useful variables used by this script :
ALPHAFOLD3_DATA=${ALPHAFOLD3_DATA}
MGNIFY_DATABASE_PATH=${ALPHAFOLD3_DATA}/mgy_clusters_2022_05.fa
ntrna_database_path=${ALPHAFOLD3_DATA}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
pdb_database_path=${ALPHAFOLD3_DATA}/pdb_2022_09_28_mmcif_files.tar
RFAM_DATABASE_PATH=${ALPHAFOLD3_DATA}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
RNA_CENTRAL_DATABASE_PATH=${ALPHAFOLD3_DATA}/rnacentral_active_seq_id_90_cov_80_linclust.fasta
SEQRES_DATABASE_PATH=${ALPHAFOLD3_DATA}/pdb_seqres_2022_09_28.fasta
SMALL_BFD_DATABASE_PATH=${ALPHAFOLD3_DATA}/bfd-first_non_consensus_sequences.fasta
UNIPROT_CLUSTER_ANNOT_DATABASE_PATH=${ALPHAFOLD3_DATA}/uniprot_all_2021_04.fa
UNIREF90_DATABASE_PATH=${ALPHAFOLD3_DATA}/uniref90_2022_05.fa

OUTDIR=`pwd`
RUN_INFERENCE=--norun_inference
RUN_DATA_PIPELINE=--norun_data_pipeline
JACKHMMER_N_CPU=8
NHMMER_N_CPU=8
JAX_COMPILATION_CACHE_DIR=/pasteur/appa/scratch


function usage() {
    echo "Usage: $(basename ${0}) [-m MODEL_DIR] [options] alphafold3_input"
    echo "Wrapper script for alphafold with default values preset"
    echo "  -h | --help                      ... display this message and exit."
    echo "  -d | --db_dir <dir>              ... Use <dir> for alphafold data location."
    echo "                                       (default ${ALPHAFOLD3_DATA})"
    echo "  -m | --model_dir <dir>           ... MANDATORY by default or when -D | --run_inference is set"
    echo "                                       (Path to the model to use for inference)"
    echo "  -j | --jackhmmer_n_cpu <int>     ... Number of CPUs to use for Jackhmmer.."
    echo "                                       (default 8)"
    echo "  -n | --nhmmer_n_cpu <int>        ... Number of CPUs to use for Nhmmer."
    echo "                                       (default 8)"
    echo "  -o | --out <dir>                 ... Use <dir> for OUTDIR."
    echo "                                       (default current working directory)"
    echo "                                       will be created if does not exist" 
    echo "  -D | --run_data_pipeline         ... Only run the data pipeline."
    echo "  -I | --run_inference             ... Only run inference pipeline"
    echo ""
    echo "alphafold3_input is "
    echo "    either the path to a single JSON file"
    echo "    either the path to a directory of JSON files"
}


#---- transform long options to short ones to be processed by getopt
for arg in "$@"; do
  shift
  case "$arg" in
    "--help")                   set -- "$@" "-h" ;;
    "--db_dir")                 set -- "$@" "-d" ;;
    "--model_dir")              set -- "$@" "-m" ;;
    "--input_json")             set -- "$@" "-i" ;;
    "--input_dir")              set -- "$@" "-d" ;;
    "--jackhmmer_n_cpu")        set -- "$@" "-j" ;;
    "--nhmmer_n_cpu")           set -- "$@" "-n" ;;
    "--out")                    set -- "$@" "-o" ;;
    "--run_inference")          set -- "$@" "-I" ;;
    "--run_data_pipeline")      set -- "$@" "-D" ;;
    *)                          set -- "$@" "$arg"
  esac
done

#---- Parse short options
OPTIND=1
while getopts "hd:m:j:n:o:ID" opt
do
  case "$opt" in
    "h") usage; exit 0 ;;
    "d") ALPHAFOLD3_DATA=${OPTARG} ;;                     # toggle DBs location to provided one
    "d") ALPHAFOLD3_DATA=${OPTARG} ;;                     # toggle DBs location to provided one
    "m") ALPHAFOLD_MODELS=${OPTARG} ;;                    # toggle models to provided one 
    "o") OUTDIR=${OPTARG} ;;                              # set outdir to provided one
    "I") RUN_INFERENCE='--run_inference'  ;;               # to run or not inference n GPU
    "D") RUN_DATA_PIPELINE='--run_data_pipeline' ;;       # to run or not data pipeline on the fold inputs.
    "j") JACKHMMER_N_CPU=${OPTARG} ;;                     # set Jackhmmer number of cpu to use
    "n") NHMMER_N_CPU=${OPTARG} ;;                        # set Nhmmer number of cpu to use
    "?") usage; exit 1 ;;
  esac
done

shift $(expr $OPTIND - 1)

#---- chek data location
if  [ ! -d ${ALPHAFOLD3_DATA} ]; then
    echo "ERROR: ${ALPHAFOLD3_DATA} No such file or directory. Exiting."
    exit 1
fi

#---- was run_pipeline or run_inference set
#     if both are not set -> run the whole workflow
if [[ "${RUN_INFERENCE}" == "--norun_inference" && "${RUN_DATA_PIPELINE}" == "--norun_data_pipeline" ]]
then
   RUN_INFERENCE="--run_inference"
   RUN_DATA_PIPELINE="--run_data_pipeline"
fi

cmd_args=''
#---- check input to toggle --input_json // --input_dir
if [ -z ${1} ]; then usage; exit 1; fi
if [ -f ${1} ]; then
    cmd_args="${cmd_args} --json_path=${1} "
elif [ -d ${1} ]; then 
    cmd_args="${cmd_args} --input_dir=${1} "
else
    echo "no valid input found"
    exit 1
fi


NV_REQUIRED='OFF'  # will be checked by alphafold.sh wrapper to toggle
                   # or not --nv apptainer option. only required by 
                   # inference step.

#---- check if we do need the models, check model_dir and model file
if [ "${RUN_INFERENCE}" == "--run_inference" ]
then
    #---- toggle apptainer NV's requirements ON
    #     will be captured by alphafold.sh wrapper
    NV_REQUIRED=ON
    #---- is model_dir set 
    if [[ ! -n "${ALPHAFOLD_MODELS}" ]]; 
    then
       echo "ERROR: Models must be set. exiting."
       exit 1
    else
        cmd_args="${cmd_args} --model_dir=${ALPHAFOLD_MODELS}"
    fi
    #---- does it exists
    if [ ! -d ${ALPHAFOLD_MODELS} ]
    then
       echo "ERROR: -m | --model_dir ${ALPHAFOLD_MODELS}. No such directory"
       exit 1
    fi
    #---- is model file available
    if [ ! -f "${ALPHAFOLD_MODELS}/af3.bin.zst" ]
    then 
           echo "ERROR: ${ALPHAFOLD_MODELS}/af3.bin.zst: AlphaFold 3 model parameters"
           exit 1
    fi
fi

#---- check OUTDIR
test -d ${OUTDIR} || mkdir -p ${OUTDIR}

cmd_args="${cmd_args} \
          --jackhmmer_n_cpu=${JACKHMMER_N_CPU} \
          --nhmmer_n_cpu=${NHMMER_N_CPU} \
          --output_dir=${OUTDIR} \
          --db_dir=${ALPHAFOLD3_DATA} \
          ${RUN_DATA_PIPELINE} \
          ${RUN_INFERENCE} 
        "
#ALPHAFOLD3_DATA $ALPHAFOLD3_DATA
#ALPHAFOLD_MODELS $ALPHAFOLD_MODELS
#OUTDIR $OUTDIR
#RUN_INFERENCE $RUN_INFERENCE
#RUN_DATA_PIPELINE $RUN_DATA_PIPELINE
#JACKHMMER_N_CPU $JACKHMMER_N_CPU
#NHMMER_N_CPU $NHMMER_N_CPU

test "x$DEBUG" = xx && cat <<EOT
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

NV_REQUIRED=${NV_REQUIRED} exec run_alphafold.py ${cmd_args}
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
EOT

NV_REQUIRED=${NV_REQUIRED} apptainer exec --nv -B /pasteur -B /local -B /opt/gensoft/data/alphafold/3.0.1 \
    -B ~/alphafold3/alphafold_custom/:/app/alphafold/ \
    -B ~/alphafold3/alphafold3_venv_custom/:/alphafold3_venv \
    /opt/gensoft/exe/alphafold/3.0.1/libexec/alphafold-3.0.1.simg \
    bash -c ". /alphafold3_venv/bin/activate && python3 /app/alphafold/run_alphafold.py ${cmd_args}"

#NV_REQUIRED=${NV_REQUIRED} apptainer exec --nv -B /pasteur -B /local -B /opt/gensoft/data/alphafold/3.0.1 \
#    -B ~/alphafold_custom/:/app/alphafold/ \
#    /opt/gensoft/exe/alphafold/3.0.1/libexec/alphafold-3.0.1.simg \
#    bash -c ". /alphafold3_venv/bin/activate && python3 /app/alphafold/run_alphafold.py ${cmd_args}"
    
#NV_REQUIRED=${NV_REQUIRED} exec run_alphafold.py ${cmd_args}