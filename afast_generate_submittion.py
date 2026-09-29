import argparse
import glob
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and submit a Slurm submission script for AlphaFast."
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
        "-b",
        "--base_path",
        default=None,
        help="Base path containing the inputs directory (Defaults to the current working directory)",
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

    # Step 1: Handle base path resolution in Python
    if args.base_path is None:
        base_dir = os.getcwd()
        print(f"Notice: No base path provided. Defaulting to current working directory: {base_dir}")
    else:
        base_dir = os.path.abspath(args.base_path)

    inputs_dir = os.path.join(base_dir, "inputs")

    # Step 2: Validate targeted workspace directories
    if not os.path.isdir(base_dir):
        print(f"Error: The resolved base path directory '{base_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(inputs_dir):
        print(f"Error: Required 'inputs' directory not found at '{inputs_dir}'. Check your files.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Count JSON inputs and calculate batch/array requirements
    json_files = glob.glob(os.path.join(inputs_dir, "*.json"))
    total_inputs = len(json_files)

    if total_inputs == 0:
        print(f"Error: No .json structural files found inside '{inputs_dir}'.", file=sys.stderr)
        sys.exit(1)

    chunk_size = 100
    max_array_idx = (total_inputs - 1) // chunk_size

    print(f"Found {total_inputs} input structures inside '{inputs_dir}'.")
    print(f"Splitting into {max_array_idx + 1} batch task(s) of up to {chunk_size} items each.")

    # Step 4: Construct the pure Bash template layout
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

if [ -z "$BASE_PATH" ]; then
    BASE_PATH="{base_dir}"
fi

if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Base directory '$BASE_PATH' does not exist."
    exit 1
fi

if [ ! -d "$BASE_PATH/inputs" ]; then
    echo "Error: Directory '$BASE_PATH/inputs' not found."
    exit 1
fi

mkdir -p "$BASE_PATH/outputs"

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Notice: Slurm Job Array mode detected. Running index: ${{SLURM_ARRAY_TASK_ID}}"

  TASK_INPUT_DIR="$SLURM_TMPDIR/inputs_task_${{SLURM_ARRAY_TASK_ID}}"
  mkdir -p "$TASK_INPUT_DIR"
  ALL_INPUTS=($(ls -1 "${{BASE_PATH}}/inputs"/*.json | sort))
  TOTAL_INPUTS=${{#ALL_INPUTS[@]}}
  CHUNK_SIZE=100
  START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
  END_IDX=$((START_IDX + CHUNK_SIZE - 1))

  if [ $END_IDX -ge $TOTAL_INPUTS ]; then
      END_IDX=$((TOTAL_INPUTS - 1))
  fi

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
echo "AFAST:      $AFAST"
echo "Running alphafast processing chunk..."

${{AFAST}}/scripts/run_alphafast.sh \\
    --input_dir "${{TASK_INPUT_DIR}}" \\
    --output_dir "${{BASE_PATH}}/outputs" \\
    --db_dir "${{AFAST}}/dbs" \\
    --weights_dir "${{AFAST}}/weights" \\
    --temp_dir "$SLURM_TMPDIR/alphafast_tmp" \\
    --container "${{AFAST}}/alphafast.sif" \\
    --jax_compilation_cache_dir "$SLURM_TMPDIR/alphafast_jax_cache" \\
    --gpu_devices 0 \\
    --num_workers 1

echo "done"
"""

    # Step 5: Save script to file system
    try:
        with open(args.output_script, "w", encoding="utf-8") as f:
            f.write(slurm_template)

        os.chmod(args.output_script, 0o755)
        print(f"Successfully generated executable Slurm script at: {args.output_script}")
    except Exception as e:
        print(f"Error writing file: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 6: Submit to Slurm via sbatch command execution
    sbatch_cmd = ["sbatch", f"--array=0-{max_array_idx}", args.output_script]

    # Forward optional base_path or custom afast arguments into the sbatch execution payload
    if args.base_path:
        sbatch_cmd.extend(["-b", base_dir])
    if args.afast != "/scratch/$USER/programs/alphafast":
        sbatch_cmd.extend(["-a", args.afast])

    print(f"To submit: {' '.join(sbatch_cmd)}")


if __name__ == "__main__":
    main()
