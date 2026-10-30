#!/usr/bin/env python3
"""
04_generate_afcyc_pipeline.py

Generates a SLURM job-array layout that screens relaxed macrocycle-target
complexes with 04_afcycdesign_filter.py.

Workflow:
  1. python 04_generate_afcyc_pipeline.py ... [--submit]
  2. bash <out-dir>/submit_afcyc.sh [ROUND]
  3. python 04b_merge_afcyc_csvs.py <out-dir> --round N --expected-tasks T
  4. python 04c_derive_target_cutoff.py <out-dir>/afcyc_filter_summary_rN.csv

The task list is <out-dir>/backbones.list (stem<TAB>path). It is copied from
<mpnn-relax-out>/backbones.list if present, otherwise built from --backbones-dir.
The number of array tasks is derived from that list, so tasks always match it.

--afcyc-workers N runs N AfCycDesign worker processes concurrently on the task's GPU
(each takes a strided shard of the task's designs, so JAX/AlphaFold compile and model
load are paid once per worker, not once per design). Each worker writes its own CSV
(--task-id 100000 + 100*task + k); the shard CSVs are merged into the usual
logs/afcyc_r<round>_task<n>.csv when all workers finish, so 04b_merge_afcyc_csvs.py
is unchanged. With N > 1 CUDA MPS is started per task unless --no-mps is given.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

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
#SBATCH --requeue

# ROUND is exported by submit_afcyc.sh (--export=ALL,ROUND=n).
set -uo pipefail
: "${ROUND:?ROUND must be exported; launch via submit_afcyc.sh}"

OUT="__OUT__"
MPNN_RELAX_OUT="__MPNN_RELAX_OUT__"
LIST="${OUT}/backbones.list"
CHUNK=__CHUNK__
WORKERS=__AFCYC_WORKERS__
USE_MPS=__USE_MPS__

TOTAL=$(wc -l < "${LIST}")
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
[ "${END}" -gt "${TOTAL}" ] && END=${TOTAL}
if [ "${START}" -gt "${TOTAL}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} is beyond the list (${TOTAL} entries); nothing to do."
    exit 0
fi

echo "=== AfCycDesign round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}, entries ${START}-${END} on $(hostname) ==="

# Build this task's job file: stem <TAB> relaxed complex for this round.
# Designs whose relaxed file is missing upstream are skipped.
JOBS_TSV="${OUT}/logs/afcyc_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.tsv"
: > "${JOBS_TSV}"
sed -n "${START},${END}p" "${LIST}" | while IFS=$'\t' read -r STEM ORIG; do
    RELAXED="${MPNN_RELAX_OUT}/${STEM}/round${ROUND}/${STEM}_r${ROUND}.pdb"
    if [ -s "${RELAXED}" ]; then
        printf '%s\t%s\n' "${STEM}" "${RELAXED}" >> "${JOBS_TSV}"
    fi
done

N_JOBS=$(wc -l < "${JOBS_TSV}")
if [ "${N_JOBS}" -eq 0 ]; then
    echo "No relaxed structures found for this chunk; skipping."
    rm -f "${JOBS_TSV}"
    exit 0
fi

# Skip if the task CSV already has a row (pass, fail or ERROR) for every job.
TASK_CSV="${OUT}/logs/afcyc_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.csv"
if [ -f "${TASK_CSV}" ]; then
    N_DONE=$(( $(wc -l < "${TASK_CSV}") - 1 ))
    if [ "${N_DONE}" -ge "${N_JOBS}" ]; then
        echo "Task CSV already has ${N_DONE} rows for ${N_JOBS} jobs; skipping."
        rm -f "${JOBS_TSV}"
        exit 0
    fi
fi

W=${WORKERS}
[ "${W}" -gt "${N_JOBS}" ] && W=${N_JOBS}

if [ "${W}" -le 1 ]; then
    echo "[*] ${N_JOBS} designs queued; launching container."
    singularity exec --nv -B __BIND_PATH__,__SCRIPT_DIR__ "__CONTAINER_SIF__" \
        python3 "__SCRIPT_DIR__/04_afcycdesign_filter.py" \
        --jobs-tsv "${JOBS_TSV}" \
        --out-dir "${OUT}" \
        --round "${ROUND}" \
        --task-id "${SLURM_ARRAY_TASK_ID}" \
        --ipae-cutoff __IPAE_CUTOFF__ \
        --rmsd-cutoff __RMSD_CUTOFF__ \
        --plddt-cutoff __PLDDT_CUTOFF__ \
        --binder-chain __BINDER_CHAIN__ \
        --target-chain __TARGET_CHAIN__
    STATUS=$?

    # Keep the job file if the worker failed, to ease debugging.
    [ "${STATUS}" -eq 0 ] && rm -f "${JOBS_TSV}"
    exit "${STATUS}"
fi

echo "[*] ${N_JOBS} designs queued; ${W} parallel worker process(es) on one GPU."

# Several JAX processes on one GPU: do not let each preallocate most of the GPU memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Optional CUDA MPS so the workers' kernels can overlap on the one GPU.
MPS_BIND=""
if [ "${USE_MPS}" -eq 1 ]; then
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
    export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${SLURM_JOB_ID}"
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    if nvidia-cuda-mps-control -d; then
        MPS_BIND="-B ${CUDA_MPS_PIPE_DIRECTORY}"
    else
        echo "[WARN] could not start the MPS daemon; continuing without MPS"
        unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
        USE_MPS=0
    fi
fi

PIDS=()
SHARD_CSVS=()
for (( k=0; k<W; k++ )); do
    SID=$(( 100000 + SLURM_ARRAY_TASK_ID * 100 + k ))
    PFX="${OUT}/logs/afcyc_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.shard${k}"
    # strided split so each shard gets a similar mix of designs
    awk -v n="${W}" -v k="${k}" '(NR-1)%n==k' "${JOBS_TSV}" > "${PFX}.tsv"
    SHARD_CSVS+=("${OUT}/logs/afcyc_r${ROUND}_task${SID}.csv")
    singularity exec --nv -B __BIND_PATH__,__SCRIPT_DIR__ ${MPS_BIND} "__CONTAINER_SIF__" \
        python3 "__SCRIPT_DIR__/04_afcycdesign_filter.py" \
        --jobs-tsv "${PFX}.tsv" \
        --out-dir "${OUT}" \
        --round "${ROUND}" \
        --task-id "${SID}" \
        --ipae-cutoff __IPAE_CUTOFF__ \
        --rmsd-cutoff __RMSD_CUTOFF__ \
        --plddt-cutoff __PLDDT_CUTOFF__ \
        --binder-chain __BINDER_CHAIN__ \
        --target-chain __TARGET_CHAIN__ > "${PFX}.log" 2>&1 &
    PIDS+=($!)
    sleep 1    # avoid all processes initialising CUDA in the same instant
done

FAIL=0
for k in "${!PIDS[@]}"; do
    wait "${PIDS[$k]}" || { echo "[WARN] AfCycDesign worker ${k} exited non-zero"; FAIL=1; }
    echo "----- AfCycDesign worker ${k} log -----"
    cat "${OUT}/logs/afcyc_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.shard${k}.log"
done

[ "${USE_MPS}" -eq 1 ] && { echo quit | nvidia-cuda-mps-control || true; }

# Merge the shard CSVs into the task CSV that 04b_merge_afcyc_csvs.py expects
# (header taken from the first non-empty shard).
EXISTING=()
for f in "${SHARD_CSVS[@]}"; do [ -s "${f}" ] && EXISTING+=("${f}"); done
if [ "${#EXISTING[@]}" -eq 0 ]; then
    echo "[ERROR] no shard CSV found (expected e.g. ${SHARD_CSVS[0]}); check how the worker names its CSV from --task-id"
    exit 1
fi
awk 'FNR==1 && NR!=1 {next} {print}' "${EXISTING[@]}" > "${TASK_CSV}.tmp" && mv "${TASK_CSV}.tmp" "${TASK_CSV}"
echo "[*] merged ${#EXISTING[@]} shard CSV(s) into ${TASK_CSV} ($(( $(wc -l < "${TASK_CSV}") - 1 )) rows for ${N_JOBS} designs)"

if [ "${FAIL}" -eq 0 ]; then
    rm -f "${EXISTING[@]}" "${JOBS_TSV}"
    for (( k=0; k<W; k++ )); do
        rm -f "${OUT}/logs/afcyc_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.shard${k}."{tsv,log}
    done
else
    echo "[WARN] keeping shard files in ${OUT}/logs for debugging / resume"
fi
exit "${FAIL}"
"""

SUBMIT_TEMPLATE = r"""#!/bin/bash
# submit_afcyc.sh -- submit AfCycDesign arrays (all rounds, or one: bash submit_afcyc.sh 3)
set -uo pipefail

OUT="__OUT__"
SLURM_FILE="${OUT}/afcyc_step.slurm"
N_ROUNDS=__N_ROUNDS__
MAX_CONC=__MAX_CONC__
N_TASKS=__N_TASKS__

if [ $# -gt 0 ]; then
    ROUND="$1"
    echo "Submitting AfCycDesign array for round ${ROUND}"
    sbatch --export=ALL,ROUND="${ROUND}" --array=1-${N_TASKS}%${MAX_CONC} "${SLURM_FILE}"
    exit 0
fi

echo "Submitting ${N_ROUNDS} rounds x ${N_TASKS} tasks (up to ${MAX_CONC} concurrent GPUs per round)"
for ROUND in $(seq 1 "${N_ROUNDS}"); do
    echo "[*] Queueing round ${ROUND}/${N_ROUNDS}"
    sbatch --export=ALL,ROUND="${ROUND}" --array=1-${N_TASKS}%${MAX_CONC} "${SLURM_FILE}"
done
"""


def fill(template, mapping):
    for k, v in mapping.items():
        template = template.replace(f"__{k}__", str(v))
    return template


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones-dir", default=None,
                    help="Only needed if <mpnn-relax-out>/backbones.list does not exist")
    ap.add_argument("--mpnn-relax-out", required=True,
                    help="Directory with the ProteinMPNN + FastRelax outputs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sif", "--colabdesign-sif", dest="sif", required=True,
                    help="Container with JAX, CUDA and ColabDesign")
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--max-backbones", type=int, default=None, help="Cap to first N list entries")
    ap.add_argument("--queue", default="gh-bio")
    ap.add_argument("--bind-path", default="/net/nfs-bio")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--max-concurrent", type=int, default=50,
                    help="Concurrent array tasks PER ROUND (all rounds are submitted together)")
    ap.add_argument("--submit", action="store_true")

    ap.add_argument("--ipae-cutoff", "--norm-ipae-cutoff", dest="ipae_cutoff", type=float, default=0.30)
    ap.add_argument("--rmsd-cutoff", type=float, default=1.5)
    ap.add_argument("--plddt-cutoff", type=float, default=0.0,
                    help="0 disables the pLDDT gate (pLDDT is still recorded)")
    ap.add_argument("--binder-chain", default="A")
    ap.add_argument("--target-chain", default="B")

    ap.add_argument("--afcyc-time", default="00:20:00")
    ap.add_argument("--afcyc-mem", default=None, help="Default: max(32, 8 x workers) GB")
    ap.add_argument("--afcyc-cpus", type=int, default=None, help="Default: max(4, workers + 2)")
    ap.add_argument("--afcyc-workers", type=int, default=1,
                    help="Concurrent AfCycDesign processes per array task on the task's GPU "
                         "(default 1 = old behaviour). Capped at the designs in the chunk; use "
                         "--chunk-size >= ~8-10 x workers so each process amortises its JAX compile.")
    ap.add_argument("--no-mps", action="store_true",
                    help="Do not start CUDA MPS when --afcyc-workers > 1")
    args = ap.parse_args()

    if args.afcyc_workers < 1:
        sys.exit("--afcyc-workers must be >= 1")
    afcyc_cpus = args.afcyc_cpus or max(4, args.afcyc_workers + 2)
    afcyc_mem = args.afcyc_mem if args.afcyc_mem is not None else f"{max(32, 8 * args.afcyc_workers)}G"
    use_mps = 1 if (args.afcyc_workers > 1 and not args.no_mps) else 0

    mpnn_relax_path = Path(args.mpnn_relax_out).resolve()
    out_path = Path(args.out_dir).resolve()
    (out_path / "logs").mkdir(parents=True, exist_ok=True)

    # Build the task list that the array tasks will index into.
    src_list = mpnn_relax_path / "backbones.list"
    if src_list.exists():
        lines = [l for l in src_list.read_text().splitlines() if l.strip()]
    else:
        if not args.backbones_dir:
            sys.exit(f"{src_list} not found; pass --backbones-dir to build a list.")
        bdir = Path(args.backbones_dir).resolve()
        pdbs = sorted(bdir.glob("*.pdb")) + sorted(bdir.glob("*.cif"))
        if not pdbs:
            sys.exit(f"No .pdb/.cif files under {bdir}")
        lines = [f"{p.stem}\t{p}" for p in pdbs]
    if args.max_backbones is not None:
        lines = lines[:args.max_backbones]
        print(f"Capping to first {len(lines)} list entries.")
    if not lines:
        sys.exit("Task list is empty.")
    (out_path / "backbones.list").write_text("\n".join(lines) + "\n")

    n_tasks = math.ceil(len(lines) / args.chunk_size)

    mapping = {
        "OUT": out_path,
        "MPNN_RELAX_OUT": mpnn_relax_path,
        "CONTAINER_SIF": Path(args.sif).resolve(),
        "BIND_PATH": args.bind_path,
        "CHUNK": args.chunk_size,
        "GPU_QUEUE": args.queue,
        "AFCYC_TIME": args.afcyc_time,
        "AFCYC_MEM": afcyc_mem,
        "AFCYC_CPUS": afcyc_cpus,
        "AFCYC_WORKERS": args.afcyc_workers,
        "USE_MPS": use_mps,
        "N_TASKS": n_tasks,
        "MAX_CONC": args.max_concurrent,
        "N_ROUNDS": args.n_rounds,
        "SCRIPT_DIR": SCRIPT_DIR,
        "IPAE_CUTOFF": args.ipae_cutoff,
        "RMSD_CUTOFF": args.rmsd_cutoff,
        "PLDDT_CUTOFF": args.plddt_cutoff,
        "BINDER_CHAIN": args.binder_chain,
        "TARGET_CHAIN": args.target_chain,
    }

    (out_path / "afcyc_step.slurm").write_text(fill(AFCYC_TEMPLATE, mapping))
    submit_script = out_path / "submit_afcyc.sh"
    submit_script.write_text(fill(SUBMIT_TEMPLATE, mapping))
    submit_script.chmod(0o755)

    print("=" * 78)
    print("AfCycDesign SLURM layout generated")
    print("=" * 78)
    print(f"Code directory       : {SCRIPT_DIR}")
    print(f"MPNN + Relax input   : {mpnn_relax_path}")
    print(f"Output directory     : {out_path}")
    print(f"Designs in list      : {len(lines)}")
    print(f"Rounds               : {args.n_rounds}")
    print(f"Array tasks / round  : {n_tasks} (chunk size {args.chunk_size})")
    print(f"Peak GPUs if all rounds submitted together: {args.n_rounds * args.max_concurrent}")
    print(f"Workers per task     : {args.afcyc_workers} (cpus {afcyc_cpus}, mem {afcyc_mem}, "
          f"MPS {'on' if use_mps else 'off'})")
    if args.afcyc_workers > 1 and args.chunk_size < 8 * args.afcyc_workers:
        print(f"  note: chunk size {args.chunk_size} gives < 8 designs per worker; a larger "
              f"--chunk-size amortises JAX compile/model load better")
    print(f"Chains               : binder={args.binder_chain} target={args.target_chain}")
    print(f"Pass rule            : iPAE < {args.ipae_cutoff} | RMSD < {args.rmsd_cutoff} A"
          + (f" | pLDDT > {args.plddt_cutoff}" if args.plddt_cutoff > 0 else " | pLDDT not gated"))
    print("Each task writes logs/afcyc_r<round>_task<n>.csv; afterwards run, per round:")
    print(f"  python {SCRIPT_DIR}/04b_merge_afcyc_csvs.py {out_path} --round N --expected-tasks {n_tasks}")
    print(f"\nLaunch all rounds : bash {submit_script}")
    print(f"Launch one round  : bash {submit_script} 3")
    print("=" * 78)

    if args.submit:
        subprocess.run(["bash", str(submit_script)], check=True)


if __name__ == "__main__":
    main()
