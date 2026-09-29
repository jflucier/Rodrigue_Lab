import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Slurm submission script for AlphaFast."
    )

    valid_gpus = [
        "h100_1g.10gb:1",
        "h100_2g.20gb:1",
        "h100_3g.40gb:1",
        "h100_80gb:1",
    ]

    # Required parameters
    parser.add_argument(
        "--account",
        required=True,
        help="Slurm account name (Required, e.g., def-rodrigu1)",
    )
    parser.add_argument(
        "--gpus",
        required=True,
        choices=valid_gpus,
        help=f"GPU resource profile (Required). Options: {', '.join(valid_gpus)}",
    )

    # Optional parameters with default values
    parser.add_argument(
        "--cpu",
        default="4",
        help="Number of CPUs per task (Default: 4)",
    )
    parser.add_argument(
        "--mem",
        default="62G",
        help="Memory requirement (Default: 62G)",
    )
    parser.add_argument(
        "--time",
        default="12:00:00",
        help="Walltime limit (Default: 12:00:00)",
    )
    parser.add_argument(
        "--afast",
        default="/scratch/$USER/programs/alphafast",
        help="Path to AlphaFast program root (Default: /scratch/$USER/programs/alphafast)",
    )

    parser.add_argument(
        "-o",
        "--output_script",
        default="submit_afast.sh",
        help="Path where the generated script will be saved (Default: submit_afast.sh)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Dynamic Bash code snippet to inject into the template fallback logic
    # If -b is missing, fallback to current working directory (PWD)
    base_path_fallback_logic = """# Step 1: Handle fallback if base path was not provided
if [ -z "$BASE_PATH" ]; then
    BASE_PATH="$(pwd)"
    echo "Notice: No base path provided. Defaulting to current working directory: $BASE_PATH"
fi"""

    slurm_template = f"""#!/bin/bash
#SBATCH --job-name=afast
#SBATCH --output=afast_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpu}
#SBATCH --mem={args.mem}
#SBATCH --gpus={args.gpus}
#SBATCH --time={args.time}
#SBATCH --account={args.account}

ml apptainer/1.3.5
ml cuda/12.2

usage() {{
    echo "Usage: $0 [-b <base_path>] [-a <afast_path>]"
    echo "  -b    Base path (Optional, defaults to current working directory)"
    echo "  -a    AFAST path (Optional, defaults to {args.afast})"
    exit 1
}}

BASE_PATH=""
AFAST="{args.afast}"

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

{base_path_fallback_logic}

# Step 2: Validate that the resolved base path directory exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: The resolved base path '$BASE_PATH' does not exist."
    exit 1
fi

# Step 3: Validate that the inputs folder inside the base path exists
if [ ! -d "$BASE_PATH/inputs" ]; then
    echo "Error: Required directory '$BASE_PATH/inputs' not found."
    exit 1
fi

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Notice: Slurm Job Array mode detected. Running index: ${{SLURM_ARRAY_TASK_ID}}"

  TASK_INPUT_DIR="$SLURM_TMPDIR/inputs_task_${{SLURM_ARRAY_TASK_ID}}"
  mkdir -p "$TASK_INPUT_DIR"
  ALL_INPUTS=($(ls -1 "${{BASE_PATH}}/inputs"/*.json | sort))
  TOTAL_INPUTS=${{#ALL_INPUTS[@]}}
  CHUNK_SIZE=100
  START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
  END_IDX=$((START_IDX + CHUNK_SIZE - 1))

  # Prevent index out of bounds on the final task chunk iteration
  if [ $END_IDX -ge $TOTAL_INPUTS ]; then
      END_IDX=$((TOTAL_INPUTS - 1))
  fi

  # Symlink the exact assigned JSON block into the temporary folder
  for ((i=START_IDX; i<=END_IDX; i++)); do
      FILE_PATH="${{ALL_INPUTS[i]}}"
      if [ -f "$FILE_PATH" ]; then
          cp "$FILE_PATH" "$TASK_INPUT_DIR/$(basename "$FILE_PATH")"
      fi
  done
else
  TASK_INPUT_DIR="$SLURM_TMPDIR/inputs_task"
  mkdir -p "$TASK_INPUT_DIR"
  cp "${{BASE_PATH}}"/inputs/* "${{TASK_INPUT_DIR}}"
fi

if [ -z "$(ls -A "$TASK_INPUT_DIR" 2>/dev/null)" ]; then
    echo "Notice: No files assigned to this array task index. Exiting cleanly."
    exit 0
fi

echo "Base Path:  $BASE_PATH"
echo "AFAST: $AFAST"
echo "Running alphafast on ${{BASE_PATH}}/inputs"
${{AFAST}}/scripts/run_alphafast.sh \\
    --input_dir "${{TASK_INPUT_DIR}}" \\
    --output_dir ${{BASE_PATH}}/outputs \\
    --db_dir ${{AFAST}}/dbs \\
    --weights_dir ${{AFAST}}/weights \\
    --temp_dir $SLURM_TMPDIR/alphafast_tmp \\
    --container ${{AFAST}}/alphafast.sif \\
    --jax_compilation_cache_dir $SLURM_TMPDIR/alphafast_jax_cache \\
    --gpu_devices 0
echo "done"
"""

    try:
        with open(args.output_script, "w", encoding="utf-8") as f:
            f.write(slurm_template)

        os.chmod(args.output_script, 0o755)
        print(
            f"Successfully generated executable Slurm script at: {args.output_script}"
        )
    except Exception as e:
        print(f"Error writing file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
