#!/usr/bin/env python
"""
03_sequence_design_cycle.py

Generates a chain of SLURM job arrays for the iterative
ProteinMPNN -> PyRosetta FastRelax (cyclized) design loop (paper Methods 2.2.2),
split into two kinds of jobs per round:

    round r:  [ MPNN array (GPU) ]  --afterany-->  [ FastRelax array (CPU) ]
              --afterany--> round r+1 MPNN array -> ...

Each array task handles a chunk of backbones (--chunk-size), so tens of
thousands of backbones don't mean tens of thousands of array tasks.
Dependencies use `afterany` and every task skips backbones whose input from
the previous round is missing, so one failed backbone doesn't cancel the
rest of the campaign.

Single combined container (ProteinMPNN + PyRosetta) is used for both steps;
only the MPNN step gets --nv.

--mpnn-workers N runs N concurrent ProteinMPNN processes per MPNN array task on
the one GPU (a single process uses only a small fraction of it). Each task's
chunk is split across the workers (strided); with N > 1 CUDA MPS is started per
task unless --no-mps is given. FastRelax is CPU-only and is unchanged.

Generated in --out-dir:
    backbones.list            stem<TAB>abs_path, one per line
    fast_relax_cyclize.xml    copied from next to this script
    03b_fastrelax_worker.py   copied from next to this script
    mpnn_step.slurm, relax_step.slurm, submit_all.sh
Per-backbone results: <out-dir>/<stem>/round<r>/<stem>_r<r>.pdb
(final designs = round <n-rounds>).

Usage:
    python 03_generate_sequence_design_cycle_pipeline.py --backbones-dir BB --out-dir OUT \
        --sif /path/combined.sif --n-rounds 4 [--submit]
"""
import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MPNN_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=mcy_mpnn
#SBATCH --partition=__GPU_QUEUE__
#SBATCH --output=__OUT__/logs/mpnn_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=__MPNN_CPUS__
#SBATCH --mem=__MPNN_MEM__
#SBATCH --time=__MPNN_TIME__

# Expects ROUND in the environment (set by submit_all.sh via --export).
set -uo pipefail

OUT="__OUT__"
LIST="${OUT}/backbones.list"
CHUNK=__CHUNK__
WORKERS=__MPNN_WORKERS__
USE_MPS=__USE_MPS__
MPNN_SCRIPT="/opt/proteinmpnn/protein_mpnn_run.py"
MPNN_WEIGHTS="/opt/proteinmpnn/vanilla_model_weights"

TOTAL=$(wc -l < "${LIST}")
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
[ "${END}" -gt "${TOTAL}" ] && END=${TOTAL}

echo "=== MPNN round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}, backbones ${START}-${END}, ${WORKERS} worker(s) ==="

# One CPU thread per worker process: many concurrent torch processes with default
# thread pools oversubscribe the node.
if [ "${WORKERS}" -gt 1 ]; then
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
fi

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

# Handles ONE backbone: all MPNN attempts for the current round. Always returns 0;
# a failed backbone is logged and skipped, as before.
process_backbone() {
    local IDX="$1"
    local PROGRESS="$2"
    local STEM ORIG PREV CURRENT_PDB ROUND_DIR TEMP MAX_ATTEMPTS ATTEMPT SUCCESS BIAS_JSON
    local FASTA_FILE MATCH_FOUND CLEAN_FASTA_CONTENT NATIVE_HEADER NATIVE_SEQ TOTAL_LINES
    local l H_IDX S_IDX H_LINE S_LINE DESIGNED_SEQ SERINE_COUNT HAS_LINEAR_PS HAS_CYCLIC_PS

    STEM=$(awk -F'\t' -v n="${IDX}" 'NR==n{print $1}' "${LIST}")
    ORIG=$(awk -F'\t' -v n="${IDX}" 'NR==n{print $2}' "${LIST}")

    if [ "${ROUND}" -eq 1 ]; then
        CURRENT_PDB="${ORIG}"
    else
        PREV=$(( ROUND - 1 ))
        CURRENT_PDB="${OUT}/${STEM}/round${PREV}/${STEM}_r${PREV}.pdb"
    fi
    if [ ! -s "${CURRENT_PDB}" ]; then
        echo "[WARN] ${STEM}: missing input ${CURRENT_PDB}, skipping"
        return 0
    fi

    ROUND_DIR="${OUT}/${STEM}/round${ROUND}"
    if ls "${ROUND_DIR}"/seqs/*.fa > /dev/null 2>&1; then
        echo "[${STEM}] round ${ROUND} MPNN output exists, skipping"
        return 0
    fi
    mkdir -p "${ROUND_DIR}"

    echo "=========================================================================="
    echo "[Progress: ${PROGRESS}] Processing backbone: ${STEM}"
    echo "=========================================================================="
    echo "[${STEM}] round ${ROUND}: ProteinMPNN Batch Constraint Solver"

    TEMP="0.0001"
    MAX_ATTEMPTS=8
    ATTEMPT=1
    SUCCESS=false

    BIAS_JSON="${ROUND_DIR}/bias_AA.json"
    echo '{"S": 0.8}' > "${BIAS_JSON}"

    while [ "${ATTEMPT}" -le "${MAX_ATTEMPTS}" ]; do
        if ! singularity exec --nv --pwd /tmp -B __BIND_PATH__ ${MPS_BIND} __SIF__ \
            python3 "${MPNN_SCRIPT}" \
            --pdb_path "${CURRENT_PDB}" \
            --pdb_path_chains "A" \
            --sampling_temp "${TEMP}" \
            --backbone_noise "0" \
            --omit_AAs "C" \
            --bias_AA_jsonl "${BIAS_JSON}" \
            --num_seq_per_target 20 \
            --path_to_model_weights "${MPNN_WEIGHTS}" \
            --out_folder "${ROUND_DIR}"; then
            echo "[WARN] ${STEM}: ProteinMPNN failed on attempt ${ATTEMPT} in round ${ROUND}"
            break
        fi

        # Locate the newly generated FASTA file
        FASTA_FILE="${ROUND_DIR}/seqs/$(basename "${CURRENT_PDB}" .pdb).fa"
        if [ ! -f "${FASTA_FILE}" ]; then
            echo "[WARN] ${STEM}: Expected fasta output ${FASTA_FILE} missing."
            break
        fi

        MATCH_FOUND=false
        CLEAN_FASTA_CONTENT=""

        # Store native reference info (first 2 lines of ProteinMPNN output)
        NATIVE_HEADER=$(awk 'NR==1' "${FASTA_FILE}")
        NATIVE_SEQ=$(awk 'NR==2' "${FASTA_FILE}")

        TOTAL_LINES=$(wc -l < "${FASTA_FILE}")
        for (( l=3; l<=${TOTAL_LINES}; l+=2 )); do
            H_IDX=$l
            S_IDX=$((l+1))

            H_LINE=$(awk "NR==${H_IDX}" "${FASTA_FILE}")
            S_LINE=$(awk "NR==${S_IDX}" "${FASTA_FILE}")

            # Isolate Chain A sequence
            DESIGNED_SEQ=$(echo "${S_LINE}" | cut -d'/' -f1)

            SERINE_COUNT=$(echo "${DESIGNED_SEQ}" | tr -cd 'S' | wc -c)
            HAS_LINEAR_PS=false; [[ "${DESIGNED_SEQ}" == *"PS"* ]] && HAS_LINEAR_PS=true
            HAS_CYCLIC_PS=false; [[ "${DESIGNED_SEQ}" == S* && "${DESIGNED_SEQ}" == *P ]] && HAS_CYCLIC_PS=true

            if [ "${SERINE_COUNT}" -eq 1 ] && [ "${HAS_LINEAR_PS}" = false ] && [ "${HAS_CYCLIC_PS}" = false ]; then
                echo "  --> Found valid sequence in batch generation: ${DESIGNED_SEQ} at temp=${TEMP}"
                # Reconstruct a clean, standard 1-sample FASTA file for PyRosetta worker compatibility
                CLEAN_FASTA_CONTENT="${NATIVE_HEADER}\n${NATIVE_SEQ}\n${H_LINE}\n${S_LINE}"
                MATCH_FOUND=true
                break
            fi
        done

        if [ "${MATCH_FOUND}" = true ]; then
            # Overwrite the multi-sequence batch file with our pristine single constrained sequence
            echo -e "${CLEAN_FASTA_CONTENT}" > "${FASTA_FILE}"
            SUCCESS=true
            break
        else
            echo "[ERROR] ${STEM}: No sequence in the 20-sample batch satisfied your criteria at T=${TEMP}."
            rm -rf "${ROUND_DIR}/seqs" "${ROUND_DIR}/scores"
        fi

        # Increase temperature to spark sidechain distribution diversity on subsequent tries
        if [ "${ATTEMPT}" -eq 1 ]; then
            TEMP="0.02"
        else
            # Scale up gradually (floating-point math via awk)
            TEMP=$(awk -v t="${TEMP}" 'BEGIN {print t + 0.02}')
        fi
        ATTEMPT=$(( ATTEMPT + 1 ))
    done

    rm -f "${BIAS_JSON}"
    if [ "${SUCCESS}" = false ]; then
        echo "[ERROR] ${STEM}: Failed composition constraint of exactly 1 Serine after ${MAX_ATTEMPTS} attempts."
        # Clean folder markers to allow manual or automated workflow re-triggers
        rm -rf "${ROUND_DIR}"
    fi
    return 0
}

# Worker W takes backbones START+W, START+W+WORKERS, ... (strided, so slow
# backbones that need several attempts are spread evenly).
worker_loop() {
    local W="$1"
    local K=0 IDX MY_TOTAL
    MY_TOTAL=$(seq $(( START + W )) "${WORKERS}" "${END}" | wc -l)
    for IDX in $(seq $(( START + W )) "${WORKERS}" "${END}"); do
        K=$(( K + 1 ))
        process_backbone "${IDX}" "w${W} ${K}/${MY_TOTAL}"
    done
    echo "worker ${W} finished ${K} backbone(s)"
}

if [ "${WORKERS}" -eq 1 ]; then
    worker_loop 0
else
    # Each worker logs to its own file (live progress: tail -f); the files are
    # appended to this log, in worker order, when all workers are done.
    PIDS=""
    for W in $(seq 0 $(( WORKERS - 1 ))); do
        WLOG="${OUT}/logs/mpnn_r${ROUND}_task${SLURM_ARRAY_TASK_ID}_w${W}.log"
        worker_loop "${W}" > "${WLOG}" 2>&1 &
        PIDS="${PIDS} $!"
    done
    echo "launched ${WORKERS} workers; live logs: ${OUT}/logs/mpnn_r${ROUND}_task${SLURM_ARRAY_TASK_ID}_w*.log"
    FAIL=0
    for P in ${PIDS}; do
        wait "${P}" || FAIL=1
    done
    for W in $(seq 0 $(( WORKERS - 1 ))); do
        WLOG="${OUT}/logs/mpnn_r${ROUND}_task${SLURM_ARRAY_TASK_ID}_w${W}.log"
        echo "------ worker ${W} ------"
        cat "${WLOG}" && rm -f "${WLOG}"
    done
    if [ "${FAIL}" -ne 0 ]; then
        echo "[ERROR] at least one worker process crashed"
    fi
fi

if [ "${USE_MPS}" -eq 1 ]; then
    echo quit | nvidia-cuda-mps-control || true
fi

echo "Done MPNN round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}, backbones ${START}-${END} "
"""

RELAX_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=mcy_relax
#SBATCH --partition=__CPU_QUEUE__
#SBATCH --output=__OUT__/logs/relax_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__RELAX_CPUS__
#SBATCH --mem=__RELAX_MEM__
#SBATCH --time=__RELAX_TIME__

# Expects ROUND in the environment (set by submit_all.sh via --export).
set -uo pipefail

OUT="__OUT__"
LIST="${OUT}/backbones.list"
CHUNK=__CHUNK__

TOTAL=$(wc -l < "${LIST}")
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
[ "${END}" -gt "${TOTAL}" ] && END=${TOTAL}

echo "=== FastRelax round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}, backbones ${START}-${END} on $(hostname) ==="

JOBS_TSV="${OUT}/logs/relax_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.tsv"
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
    [ -s "${RELAXED}" ] && { echo "[${STEM}] round ${ROUND} already relaxed, skipping"; continue; }

    FASTA=$(find "${ROUND_DIR}/seqs" -name "*.fa" 2>/dev/null | head -n 1)
    if [ -z "${FASTA}" ] || [ ! -s "${CURRENT_PDB}" ]; then
        echo "[WARN] ${STEM}: no MPNN output or missing input for round ${ROUND}, skipping"
        continue
    fi
    printf '%s\t%s\t%s\n' "${CURRENT_PDB}" "${FASTA}" "${RELAXED}" >> "${JOBS_TSV}"
done

if [ -s "${JOBS_TSV}" ]; then
    singularity exec --pwd /tmp -B __BIND_PATH__ __SIF__ \
        python3 "${OUT}/03b_fastrelax_worker.py" \
        --jobs-tsv "${JOBS_TSV}" \
        --xml "${OUT}/fast_relax_cyclize.xml" \
        --isolate --debug
else
    echo "Nothing to relax in this chunk."
fi
"""

SUBMIT_TEMPLATE = r"""#!/bin/bash
# Submits the full chain: per round, MPNN array (GPU) then FastRelax array (CPU).
set -euo pipefail
cd "__OUT__"

N_TASKS=__N_TASKS__
MAXC=__MAX_CONC__
PREV=""

for R in $(seq 1 __N_ROUNDS__); do
    DEP=()
    [ -n "${PREV}" ] && DEP=(--dependency="afterany:${PREV}")

    MPNN=$(sbatch --parsable "${DEP[@]}" --array="1-${N_TASKS}%${MAXC}" \
        --export=ALL,ROUND=${R} mpnn_step.slurm | cut -d';' -f1)
    echo "round ${R}: MPNN  array job ${MPNN}"

    RELAX=$(sbatch --parsable --dependency="afterany:${MPNN}" --array="1-${N_TASKS}%${MAXC}" \
        --export=ALL,ROUND=${R} relax_step.slurm | cut -d';' -f1)
    echo "round ${R}: RELAX array job ${RELAX}"
    PREV=${RELAX}
done
echo "Submitted. Final designs: __OUT__/<stem>/round__N_ROUNDS__/<stem>_r__N_ROUNDS__.pdb"
"""


def fill(template, mapping):
    for k, v in mapping.items():
        template = template.replace(f"__{k}__", str(v))
    return template


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sif", "--proteinmpnn-sif", dest="sif", required=True,
                    help="Combined SIF containing ProteinMPNN and PyRosetta")
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--max-backbones", type=int, default=None)
    ap.add_argument("--queue", default="gh-bio", help="GPU partition (MPNN step)")
    ap.add_argument("--cpu-queue", default=None,
                    help="Partition for FastRelax (default: same as --queue). Must be the "
                         "same CPU architecture the Rosetta/PyRosetta build in the SIF targets.")
    ap.add_argument("--bind-path", default="/net/nfs-bio")
    ap.add_argument("--chunk-size", type=int, default=20,
                    help="Backbones handled per array task (default 20)")
    ap.add_argument("--max-concurrent", type=int, default=50,
                    help="Max simultaneously running array tasks per array (default 50)")
    ap.add_argument("--mpnn-time", default="02:00:00")
    ap.add_argument("--mpnn-mem", default=None, help="Default: max(16, 4 x workers) GB")
    ap.add_argument("--mpnn-cpus", type=int, default=None, help="Default: max(2, workers)")
    ap.add_argument("--mpnn-workers", type=int, default=1,
                    help="Concurrent ProteinMPNN processes sharing the GPU inside each MPNN "
                         "array task (default 1 = previous behavior)")
    ap.add_argument("--no-mps", action="store_true",
                    help="Do not start CUDA MPS when --mpnn-workers > 1")
    ap.add_argument("--relax-time", default="08:00:00")
    ap.add_argument("--relax-mem", default="8G")
    ap.add_argument("--relax-cpus", type=int, default=1)
    ap.add_argument("--submit", action="store_true", help="Run submit_all.sh after generating")
    args = ap.parse_args()

    if args.mpnn_workers < 1:
        sys.exit("--mpnn-workers must be >= 1")
    mpnn_cpus = args.mpnn_cpus or max(2, args.mpnn_workers)
    mpnn_mem = args.mpnn_mem or f"{max(16, 4 * args.mpnn_workers)}G"
    use_mps = 1 if (args.mpnn_workers > 1 and not args.no_mps) else 0
    if args.chunk_size < 4 * args.mpnn_workers:
        print(f"[NOTE] --chunk-size {args.chunk_size} gives each of {args.mpnn_workers} workers "
              f"only ~{args.chunk_size / args.mpnn_workers:.1f} backbones per task; consider a "
              f"larger --chunk-size (e.g. {10 * args.mpnn_workers}).")

    backbones_path = Path(args.backbones_dir).resolve()
    out_path = Path(args.out_dir).resolve()
    (out_path / "logs").mkdir(parents=True, exist_ok=True)

    pdbs = sorted(backbones_path.glob("*.pdb")) + sorted(backbones_path.glob("*.cif"))
    if not pdbs:
        sys.exit(f"No .pdb/.cif files found under {backbones_path}")
    if args.max_backbones is not None:
        pdbs = pdbs[:args.max_backbones]
        print(f"Capping to the first {len(pdbs)} backbones.")

    stems = [p.stem for p in pdbs]
    if len(set(stems)) != len(stems):
        sys.exit("Duplicate file stems among backbones (a .pdb and .cif with the same name?).")

    source_xml = HERE / "fast_relax_cyclize.xml"
    worker_src = HERE / "03b_fastrelax_worker.py"
    for f in (source_xml, worker_src):
        if not f.exists():
            sys.exit(f"CRITICAL: required file not found next to this script: {f}")
    shutil.copy(source_xml, out_path / "fast_relax_cyclize.xml")
    shutil.copy(worker_src, out_path / "03b_fastrelax_worker.py")

    (out_path / "backbones.list").write_text(
        "".join(f"{p.stem}\t{p.resolve()}\n" for p in pdbs))

    n_tasks = math.ceil(len(pdbs) / args.chunk_size)
    common = {
        "OUT": out_path, "SIF": Path(args.sif).resolve(), "BIND_PATH": args.bind_path,
        "CHUNK": args.chunk_size,
        "GPU_QUEUE": args.queue, "CPU_QUEUE": args.cpu_queue or args.queue,
        "MPNN_TIME": args.mpnn_time, "MPNN_MEM": mpnn_mem, "MPNN_CPUS": mpnn_cpus,
        "MPNN_WORKERS": args.mpnn_workers, "USE_MPS": use_mps,
        "RELAX_TIME": args.relax_time, "RELAX_MEM": args.relax_mem, "RELAX_CPUS": args.relax_cpus,
        "N_TASKS": n_tasks, "MAX_CONC": args.max_concurrent, "N_ROUNDS": args.n_rounds,
    }
    (out_path / "mpnn_step.slurm").write_text(fill(MPNN_TEMPLATE, common))
    (out_path / "relax_step.slurm").write_text(fill(RELAX_TEMPLATE, common))
    submit = out_path / "submit_all.sh"
    submit.write_text(fill(SUBMIT_TEMPLATE, common))
    submit.chmod(0o755)

    print("\n" + "=" * 78)
    print(" SLURM job chain generated")
    print("=" * 78)
    print(f"Backbones: {len(pdbs)}  |  chunk size: {args.chunk_size}  |  array tasks/array: {n_tasks}")
    print(f"Rounds: {args.n_rounds}  ->  {2 * args.n_rounds} arrays (MPNN on {args.queue}, "
          f"relax on {args.cpu_queue or args.queue})")
    print(f"MPNN workers per task: {args.mpnn_workers} (cpus {mpnn_cpus}, mem {mpnn_mem}, "
          f"MPS {'on' if use_mps else 'off'})")
    print(f"Files in: {out_path}")
    print(f"\nSubmit with:\n  bash {submit}")
    print("=" * 78)

    if args.submit:
        subprocess.run(["bash", str(submit)], check=True)


if __name__ == "__main__":
    main()
