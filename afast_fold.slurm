#!/bin/bash
#SBATCH --job-name=afast
#SBATCH --output=afast_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=62G
#SBATCH --gpus=h100_3g.40gb:1
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

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Notice: Slurm Job Array mode detected. Running index: ${SLURM_ARRAY_TASK_ID}"

  TASK_INPUT_DIR="$SLURM_TMPDIR/inputs_task_${SLURM_ARRAY_TASK_ID}"
  mkdir -p "$TASK_INPUT_DIR"
  ALL_INPUTS=($(ls -1 "${BASE_PATH}/inputs"/*.json | sort))
  TOTAL_INPUTS=${#ALL_INPUTS[@]}
  CHUNK_SIZE=100
  START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
  END_IDX=$((START_IDX + CHUNK_SIZE - 1))

  # Prevent index out of bounds on the final task chunk iteration
  if [ $END_IDX -ge $TOTAL_INPUTS ]; then
      END_IDX=$((TOTAL_INPUTS - 1))
  fi

  # Symlink the exact assigned JSON block into the temporary folder
  for ((i=START_IDX; i<=END_IDX; i++)); do
      FILE_PATH="${ALL_INPUTS[i]}"
      if [ -f "$FILE_PATH" ]; then
          cp "$FILE_PATH" "$TASK_INPUT_DIR/$(basename "$FILE_PATH")"
      fi
  done
else
  TASK_INPUT_DIR="$SLURM_TMPDIR/inputs_task"
  mkdir -p "$TASK_INPUT_DIR"
  cp "${BASE_PATH}"/inputs/* "${TASK_INPUT_DIR}"
fi

if [ -z "$(ls -A "$TASK_INPUT_DIR" 2>/dev/null)" ]; then
    echo "Notice: No files assigned to this array task index. Exiting cleanly."
    exit 0
fi

echo "Base Path:  $BASE_PATH"
echo "AFAST: $AFAST"
echo "Running alphafast on ${BASE_PATH}/inputs"
${AFAST}/scripts/run_alphafast.sh \
    --input_dir "${TASK_INPUT_DIR}" \
    --output_dir ${BASE_PATH}/outputs \
    --db_dir ${AFAST}/dbs \
    --weights_dir ${AFAST}/weights \
    --temp_dir $SLURM_TMPDIR/alphafast_tmp \
    --container ${AFAST}/alphafast.sif \
    --jax_compilation_cache_dir $SLURM_TMPDIR/alphafast_jax_cache \
    --gpu_devices 0
echo "done"