#!/bin/bash
#SBATCH --job-name=dorado_gh200
#SBATCH --partition=gh-bio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --account def-rodrigu1

### GPU ALLOCATION ###
#SBATCH --gres=gpu:1

### CPU & MEMORY ALLOCATION ###
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=dorado_%j.log

##### how to call:
#sbatch dorado_basecall.slurm \
#--pod5_dir /net/nfs-bio/home/def-rodrigu1/data/brutes/blob_promethion_13_ap_2026/pod5/ \
#--out_dir $PWD/blob_promethion_13_ap_2026/
#
#additionnal parameters:
#--model name of model to use
#--model_dir path to directory where models are located
#--rclone path to rclone exe


# 1. Optimise CPU multi-threading for internal processes
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Parsing command line options
POD5_DIR=""
OUT_DIR=""
DORADO_MODEL="sup"
MODELS_DIR="/tank/def-rodrigu1/programs/dorado/models/"
RCLONE_EXE="/tank/def-rodrigu1/programs/rclone-v1.74.2-linux-arm64/rclone"

TEMP=$(getopt -o i:o:m:d:r: --long pod5_dir:,out_dir:,model:,model_dir:,rclone: -n 'dorado_basecall.slurm' -- "$@")
eval set -- "$TEMP"

while true ; do
    case "$1" in
        -i|--pod5_dir) POD5_DIR="$2" ; shift 2 ;;
        -o|--out_dir) OUT_DIR="$2" ; shift 2 ;;
        -m|--model) DORADO_MODEL="$2" ; shift 2 ;;
        -d|--model_dir) MODELS_DIR="$2" ; shift 2 ;;
        -r|--rclone) RCLONE_EXE="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Invalid Option" ; exit 1 ;;
    esac
done

# Ensure that SETTINGS is not empty
if [ -z "$POD5_DIR" ]; then
    echo "Error: The -i or --pod5_dir option is required."
    exit 1
fi
if [ -z "$OUT_DIR" ]; then
    echo "Error: The -o or --out_dir option is required."
    exit 1
fi

echo "##### INPUT PARAMETERS #####"
echo "# POD5_DIR=${POD5_DIR}"
echo "# OUT_DIR=${OUT_DIR}"
echo "# DORADO_MODEL=${DORADO_MODEL}"
echo "# MODELS_DIR=${MODELS_DIR}"
echo "# RCLONE_EXE=${RCLONE_EXE}"
echo "############################"

##POD5_DIR="/net/nfs-bio/home/def-rodrigu1/data/brutes/blob_promethion_13_ap_2026/pod5/"
#POD5_DIR="/net/nfs-bio/fast2/def-rodrigu1/basecalls/test_data/"
#OUT_DIR="/net/nfs-bio/fast2/def-rodrigu1/basecalls/20260528_promethion_13_ap_2026"
#DORADO_MODEL="sup"
#MODELS_DIR="/tank/def-rodrigu1/programs/dorado/models/"
#SAMTOOLS_EXE=/tank/def-rodrigu1/programs/samtools/samtools
#RCLONE_EXE=/tank/def-rodrigu1/programs/rclone-v1.74.2-linux-arm64/rclone

mkdir -p ${OUT_DIR}

echo "data transfert"
${RCLONE_EXE} copy \
${POD5_DIR} $SLURM_TMPDIR/data/ \
--multi-thread-streams=4 \
--ignore-checksum \
--copy-links \
--log-level=INFO \
--stats=10s \
--stats-one-line \
--buffer-size=4G

# Define a clean, unique file for this specific Slurm job run
CURRENT_BAM="${OUT_DIR}/job_${SLURM_JOB_ID}.bam"

# Look for any previous BAM files from earlier preempted attempts
PREVIOUS_BAMS=($(ls ${OUT_DIR}/job_*.bam 2>/dev/null))

RESUME_PARAM=""
if [ ${#PREVIOUS_BAMS[@]} -gt 0 ]; then
    # Target the most recent file as the resume checkpoint
    LATEST_PREVIOUS="${PREVIOUS_BAMS[-1]}"
    echo "Found existing checkpoint: ${LATEST_PREVIOUS}. Resuming cleanly..."
    RESUME_PARAM="--resume-from ${LATEST_PREVIOUS}"
fi

echo "Launching Dorado..."
# 2. Run Dorado (Outputting directly to a clean binary BAM stream)
stdbuf -oL -eL dorado basecaller -v ${DORADO_MODEL} $SLURM_TMPDIR/data/ \
    --models-directory ${MODELS_DIR} \
    --device cuda:0 \
    --batchsize 1536 \
    ${RESUME_PARAM} \
    > ${CURRENT_BAM}

if [ $? -eq 0 ]; then
    echo "Basecalling completed successfully. Finalizing production file..."
    # Rename the comprehensive current file to the final target name
    mv ${CURRENT_BAM} ${OUT_DIR}/combined_sup_reads.bam
    # Safely clear old intermediate job blocks to save cluster space
    rm -f ${OUT_DIR}/job_*.bam
    echo "Pipeline complete. Final file saved at: ${OUT_DIR}/combined_sup_reads.bam"
fi
