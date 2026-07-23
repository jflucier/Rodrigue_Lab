#!/bin/bash
#SBATCH --job-name=afast
#SBATCH --output=afast_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62G
#SBATCH --gpus=h100:1
#SBATCH --time=48:00:00
#SBATCH --account=def-rodrigu1

ml apptainer/1.3.5
ml cuda/12.2

usage() {
    echo "Usage: $0 -b <base_path> [-a <afast_path>]"
    echo "  -b    Base path (Required, must exist)"
    echo "  -a    AFAST path (Optional, defaults to /scratch/\$USER/programs/alphafast)"
    exit 1
}

BASE_PATH=""
AFAST="/scratch/$USER/programs/alphafast"

PARSED_OPTIONS=$(getopt -o "b:a:" --long "base-path:,afast:" -- "$@")
if [ $? -ne 0 ]; then
    usage
fi

eval set -- "$PARSED_OPTIONS"
while true; do
    case "$1" in
        -b|--base_path) BASE_PATH="$2" ; shift 2 ;;
        -a|--afast) AFAST="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Invalid Option" ; exit 1 ;;
    esac
done

# Step 1: Validate that base path was provided
if [ -z "$BASE_PATH" ]; then
    echo "Error: Base path (-b or --base-path) is required."
    usage
fi

# Step 2: Validate that the provided base path exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: The provided base path '$BASE_PATH' does not exist."
    exit 1
fi

echo "Base Path:  $BASE_PATH"
echo "AFAST: $AFAST"
echo "Running alphafast on ${BASE_PATH}/inputs"
${AFAST}/scripts/run_alphafast.sh \
    --input_dir ${BASE_PATH}/inputs \
    --output_dir ${BASE_PATH}/outputs \
    --db_dir ${AFAST}/dbs \
    --weights_dir ${AFAST}/weights \
    --temp_dir $SLURM_TMPDIR/alphafast_tmp \
    --container ${AFAST}/alphafast.sif \
    --jax_compilation_cache_dir $SLURM_TMPDIR/alphafast_jax_cache \
    --gpu_devices 0
echo "done"