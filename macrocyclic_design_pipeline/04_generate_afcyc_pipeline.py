#!/usr/bin/env python3
"""
04_generate_afcyc_pipeline.py

Generates a standalone SLURM Job Array layout to screen macrocycle-target
complex architectures natively across cluster Grace Hopper GPU hardware node slots.
Features complete strict residue-aware validation resume safety controls.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

# --- SLURM WORKER ARRAY STRING TEMPLATE ---
AFCYC_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=mcy_afcyc
#SBATCH --partition=__GPU_QUEUE__
#SBATCH --output=__OUT__/logs/afcyc_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=__AFCYC_CPUS__
#SBATCH --mem=__AFCYC_MEM__
#SBATCH --time=__AFCYC_TIME__

# Expects ROUND in the environment (set by submit_all.sh via --export).
set -uo pipefail

OUT="__OUT__"
LIST="${OUT}/backbones.list"
CHUNK=__CHUNK__
CONTAINER_SIF="__SIF__"
GLOBAL_CSV="${OUT}/logs/afcyc_filter_summary_r${ROUND}.csv"

TOTAL=$(wc -l < "${LIST}")
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
[ "${END}" -gt "${TOTAL}" ] && END=${TOTAL}

echo "=== AfCycDesign Prediction round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}, backbones ${START}-${END} ==="

# Define a unique worker-specific job file to prevent cross-talk on parallel threads
JOBS_TSV="${OUT}/logs/afcyc_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.tsv"
: > "${JOBS_TSV}"

for IDX in $(seq "${START}" "${END}"); do
    STEM=$(awk -F'\t' -v n="${IDX}" 'NR==n{print $1}' "${LIST}")
    ORIG=$(awk -F'\t' -v n="${IDX}" 'NR==n{print $2}' "${LIST}")

    if [ "${ROUND}" -eq 1 ]; then
        CURRENT_PDB="${ORIG}"
    else
        PREV=$(( ROUND - 1 ))
        CURRENT_PDB="${OUT}/${STEM}/round${PREV}/${STEM}_r${PREV}.pdb"
    fi

    ROUND_DIR="${OUT}/${STEM}/round${ROUND}"
    RELAXED="${ROUND_DIR}/${STEM}_r${ROUND}.pdb"
    FASTA=$(find "${ROUND_DIR}/seqs" -name "*.fa" 2>/dev/null | head -n 1)

    # Skip compilation steps if upstream files are completely missing
    if [ -z "${FASTA}" ] || [ ! -s "${CURRENT_PDB}" ] || [ ! -s "${RELAXED}" ]; then
        continue
    fi

    # Append the valid path triplets into our current worker execution tracking file
    printf '%s\t%s\t%s\n' "${CURRENT_PDB}" "${FASTA}" "${RELAXED}" >> "${JOBS_TSV}"
done

# If our worker file mapped active remaining structures, fire up the GPU container
if [ -s "${JOBS_TSV}" ]; then
    echo "[*] Task assigned ${JOBS_TSV} contains valid design targets. Launching Grace Hopper container..."

    singularity exec --nv -B __BIND_PATH__ "${CONTAINER_SIF}" \
        python3 "__OUT__/04_afcycdesign_filter.py" \
        --jobs-tsv "${JOBS_TSV}" \
        --out-csv "${GLOBAL_CSV}" \
        --norm-ipae-cutoff __IPAE_CUTOFF__ \
        --rmsd-cutoff __RMSD_CUTOFF__ \
        --plddt-cutoff __PLDDT_CUTOFF__

    # Clean temporary chunk mapping data after thread completes cleanly
    rm -f "${JOBS_TSV}"
else
    echo "All macrocycle structures inside chunk segment ${START}-${END} already logged or skipped. Skipping task execution."
    rm -f "${JOBS_TSV}"
fi
"""

# --- SUBMITTER CHAIN STRING TEMPLATE ---
SUBMIT_TEMPLATE = r"""#!/bin/bash
# submit_afcyc_round.sh
# Automated submission driver to fire up screening passes
set -uo pipefail

ROUND="${1:-1}"
echo "Submitting AfCycDesign array evaluation chain loop for Round ${ROUND}..."

sbatch --export=ALL,ROUND="${ROUND}" \
       --array=1-__N_TASKS__%__MAX_CONC__ \
       "__OUT__/afcyc_step.slurm"
"""


def fill(template, mapping):
    for k, v in mapping.items():
        template = template.replace(f"__{k}__", str(v))
    return template


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones-dir", required=True, help="Directory containing origin validation structural tracks")
    ap.add_argument("--out-dir", required=True, help="Centralized pipeline working array output tree path")
    ap.add_argument("--sif", "--colabdesign-sif", dest="sif", required=True,
                    help="SIF container containing JAX, CUDA, and ColabDesign libraries")
    ap.add_argument("--round", type=int, default=1,
                    help="Active evaluation round tracker index targeting metrics sheets")
    ap.add_argument("--max-backbones", type=int, default=None, help="Cap file evaluations to first N entries")
    ap.add_argument("--queue", default="gh-bio", help="Grace Hopper GPU cluster partition queue selector")
    ap.add_argument("--bind-path", default="/net/nfs-bio", help="External database mount directory hooks")
    ap.add_argument("--chunk-size", type=int, default=20, help="Structural files crunched per node sequence thread")
    ap.add_argument("--max-concurrent", type=int, default=50,
                    help="Maximum concurrent hardware worker nodes array limit")

    # Threshold Configuration Parameters
    ap.add_argument("--norm-ipae-cutoff", type=float, default=0.30,
                    help="Target-specific screening threshold parameter")
    ap.add_argument("--rmsd-cutoff", type=float, default=1.5, help="Maximum structural deviation parameter")
    ap.add_argument("--plddt-cutoff", type=float, default=0.80, help="Minimum prediction confidence boundary")

    # Resource Allocations
    ap.add_argument("--afcyc-time", default="12:00:00")
    ap.add_argument("--afcyc-mem", default="32G")
    ap.add_argument("--afcyc-cpus", type=int, default=4)
    ap.add_argument("--submit", action="store_false", help="Launch orchestration tasks instantly upon file printout")
    args = ap.parse_args()

    backbones_path = Path(args.backbones_dir).resolve()
    out_path = Path(args.out_dir).resolve()
    (out_path / "logs").mkdir(parents=True, exist_ok=True)

    pdbs = sorted(backbones_path.glob("*.pdb")) + sorted(backbones_path.glob("*.cif"))
    if not pdbs:
        sys.exit(f"No structural design coordinates (.pdb/.cif) discovered under: {backbones_path}")
    if args.max_backbones is not None:
        pdbs = pdbs[:args.max_backbones]
        print(f"Capping dataset evaluation queue to first {len(pdbs)} backbones.")

    # Verify or write the master listing file matching your generator configuration
    backbones_list_file = out_path / "backbones.list"
    if not backbones_list_file.exists():
        backbones_list_file.write_text("".join(f"{p.stem}\t{p.resolve()}\n" for p in pdbs))

    n_tasks = math.ceil(len(pdbs) / args.chunk_size)

    mapping = {
        "OUT": out_path,
        "SIF": Path(args.sif).resolve(),
        "BIND_PATH": args.bind_path,
        "CHUNK": args.chunk_size,
        "GPU_QUEUE": args.queue,
        "AFCYC_TIME": args.afcyc_time,
        "AFCYC_MEM": args.afcyc_mem,
        "AFCYC_CPUS": args.afcyc_cpus,
        "N_TASKS": n_tasks,
        "MAX_CONC": args.max_concurrent,

        # Injected Custom Threshold Rules
        "IPAE_CUTOFF": args.norm_ipae_cutoff,
        "RMSD_CUTOFF": args.rmsd_cutoff,
        "PLDDT_CUTOFF": args.plddt_cutoff,
    }

    # Write files out cleanly to your production cluster directories
    (out_path / "afcyc_step.slurm").write_text(fill(AFCYC_TEMPLATE, mapping))

    submit_script = out_path / "submit_afcyc.sh"
    submit_script.write_text(fill(SUBMIT_TEMPLATE, mapping))
    submit_script.chmod(0o755)

    print("\n" + "=" * 78)
    print(" 🚀 AfCycDesign SLURM Screening Grid Cluster Layout Generated Successfully")
    print("=" * 78)
    print(f"Total Structural Targets Found : {len(pdbs)}")
    print(f"Task Segments Chunk Size       : {args.chunk_size}")
    print(f"Calculated Total Array Tasks   : {n_tasks}")
    print(f"GPU Execution Node Partition   : {args.queue}")
    print(
        f"Screening Metrics Setup Rules  : Norm iPAE < {args.norm_ipae_cutoff} | RMSD < {args.rmsd_cutoff}A | pLDDT > {args.plddt_cutoff}")
    print(f"Files written completely to    : {out_path}")
    print(f"\nLaunch screening passes with commands:\n  bash {submit_script} {args.round}")
    print("=" * 78)

    if args.submit:
        subprocess.run(["bash", str(submit_script), str(args.round)], check=True)


if __name__ == "__main__":
    main()
