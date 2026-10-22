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
task unless --no-mps is given. Workers pull backbones from a dynamic queue (a worker
that finishes immediately starts the next backbone of the chunk).

Per backbone, ProteinMPNN is launched ONCE and samples the whole --mpnn-temps ladder
(--num-seq-per-temp samples each, in batches of --mpnn-batch-size); the first sequence
meeting the composition constraints (lowest temperature first) is kept.

--relax-cpus N runs N concurrent FastRelax worker processes per relax array task
(each takes a strided shard of the chunk's jobs list; one thread each).

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
#SBATCH --requeue

# Expects ROUND in the environment (set by submit_all.sh via --export).
set -uo pipefail

OUT="__OUT__"
LIST="${OUT}/backbones.list"
CHUNK=__CHUNK__
WORKERS=__MPNN_WORKERS__
TEMPS="__MPNN_TEMPS__"
NSEQ=__NUM_SEQ__
BATCH=__MPNN_BATCH__
USE_MPS=__USE_MPS__
MPNN_SCRIPT="/opt/proteinmpnn/protein_mpnn_run.py"
MPNN_WEIGHTS="/opt/proteinmpnn/vanilla_model_weights"

TOTAL=$(wc -l < "${LIST}")
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
[ "${END}" -gt "${TOTAL}" ] && END=${TOTAL}

echo "=== MPNN round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}, backbones ${START}-${END}, ${WORKERS} worker(s) ==="

# Container image: optionally run from node-local scratch ($SLURM_TMPDIR) instead of
# mounting the shared-filesystem .sif for every launch (squashfuse + network reads
# dominated startup with many concurrent launches).
#   none    : use the .sif as given
#   copy    : copy the .sif to local scratch (still a squashfs mount, but local reads)
#   sandbox : unpack the .sif to a directory on local scratch (no squashfuse at all)
IMG="__SIF__"
IMG_MODE="__IMG_MODE__"
if [ "${IMG_MODE}" != "none" ]; then
    if [ -z "${SLURM_TMPDIR:-}" ] || [ ! -d "${SLURM_TMPDIR}" ]; then
        echo "[WARN] SLURM_TMPDIR not set, using the original image"
    else
        LOCAL_ROOT="${SLURM_TMPDIR}/img_${SLURM_JOB_ID}"
        mkdir -p "${LOCAL_ROOT}"
        T0=$(date +%s)
        if [ "${IMG_MODE}" = "copy" ]; then
            cp "__SIF__" "${LOCAL_ROOT}/image.sif" && IMG="${LOCAL_ROOT}/image.sif"
        else
            singularity build --sandbox "${LOCAL_ROOT}/sandbox" "__SIF__" > "${LOCAL_ROOT}/build.log" 2>&1 \
                && IMG="${LOCAL_ROOT}/sandbox"
        fi
        if [ "${IMG}" = "__SIF__" ]; then
            echo "[WARN] local image setup (${IMG_MODE}) failed, using the original image"
            [ -f "${LOCAL_ROOT}/build.log" ] && tail -5 "${LOCAL_ROOT}/build.log"
        fi
        echo "image: ${IMG} (mode ${IMG_MODE}, setup $(( $(date +%s) - T0 )) s)"
    fi
fi

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

# Handles ONE backbone: a SINGLE ProteinMPNN call samples every temperature of the
# ladder (TEMPS) in one process (one container/CUDA/model start instead of up to 8),
# then the first sequence meeting the composition constraints is taken. The FASTA is
# ordered by temperature, so "first valid" = lowest temperature that has a valid
# sequence, exactly as the old sequential retry ladder did. Always returns 0; a failed
# backbone is logged and skipped.
process_backbone() {
    local IDX="$1"
    local PROGRESS="$2"
    local STEM ORIG PREV CURRENT_PDB ROUND_DIR SUCCESS BIAS_JSON FASTA_FILE
    local NATIVE_HEADER NATIVE_SEQ H_LINE S_LINE DESIGNED_SEQ S_ONLY SERINE_COUNT
    local HAS_LINEAR_PS HAS_CYCLIC_PS l
    local -a FA

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
    echo "[${STEM}] round ${ROUND}: ProteinMPNN single call, temperatures: ${TEMPS}"

    SUCCESS=false
    BIAS_JSON="${ROUND_DIR}/bias_AA.json"
    echo '{"S": 0.8}' > "${BIAS_JSON}"

    if ! singularity exec --nv --pwd /tmp -B __BIND_PATH__ ${MPS_BIND} "${IMG}" \
        python3 "${MPNN_SCRIPT}" \
        --pdb_path "${CURRENT_PDB}" \
        --pdb_path_chains "A" \
        --sampling_temp "${TEMPS}" \
        --backbone_noise "0" \
        --omit_AAs "C" \
        --bias_AA_jsonl "${BIAS_JSON}" \
        --num_seq_per_target "${NSEQ}" \
        --batch_size "${BATCH}" \
        --path_to_model_weights "${MPNN_WEIGHTS}" \
        --out_folder "${ROUND_DIR}"; then
        echo "[WARN] ${STEM}: ProteinMPNN failed in round ${ROUND}"
    else
        FASTA_FILE="${ROUND_DIR}/seqs/$(basename "${CURRENT_PDB}" .pdb).fa"
        if [ ! -f "${FASTA_FILE}" ]; then
            echo "[WARN] ${STEM}: Expected fasta output ${FASTA_FILE} missing."
        else
            # Read the FASTA once. Records: [native header, native seq, (header, seq) x samples],
            # samples ordered by temperature.
            mapfile -t FA < "${FASTA_FILE}"
            NATIVE_HEADER="${FA[0]}"
            NATIVE_SEQ="${FA[1]}"
            for (( l=2; l+1<${#FA[@]}; l+=2 )); do
                H_LINE="${FA[l]}"
                S_LINE="${FA[l+1]}"

                # Isolate Chain A sequence
                DESIGNED_SEQ="${S_LINE%%/*}"

                S_ONLY="${DESIGNED_SEQ//[^S]/}"
                SERINE_COUNT=${#S_ONLY}
                HAS_LINEAR_PS=false; [[ "${DESIGNED_SEQ}" == *"PS"* ]] && HAS_LINEAR_PS=true
                HAS_CYCLIC_PS=false; [[ "${DESIGNED_SEQ}" == S* && "${DESIGNED_SEQ}" == *P ]] && HAS_CYCLIC_PS=true

                if [ "${SERINE_COUNT}" -eq 1 ] && [ "${HAS_LINEAR_PS}" = false ] && [ "${HAS_CYCLIC_PS}" = false ]; then
                    echo "  --> Found valid sequence: ${DESIGNED_SEQ} (${H_LINE%%,*}, record $(( (l - 2) / 2 + 1 )) of $(( (${#FA[@]} - 2) / 2 )))"
                    # Reduce to a clean, standard 1-sample FASTA for the PyRosetta worker
                    printf '%s\n%s\n%s\n%s\n' "${NATIVE_HEADER}" "${NATIVE_SEQ}" "${H_LINE}" "${S_LINE}" > "${FASTA_FILE}"
                    SUCCESS=true
                    break
                fi
            done
            if [ "${SUCCESS}" = false ]; then
                echo "[WARN] ${STEM}: none of the $(( (${#FA[@]} - 2) / 2 )) sampled sequences satisfied the constraints."
            fi
        fi
    fi

    rm -f "${BIAS_JSON}"
    if [ "${SUCCESS}" = false ]; then
        echo "[ERROR] ${STEM}: Failed composition constraint of exactly 1 Serine (temperatures ${TEMPS})."
        # Clean folder markers to allow manual or automated workflow re-triggers
        rm -rf "${ROUND_DIR}"
    fi
    return 0
}

# Dynamic work queue: keep WORKERS backbones in flight; as soon as one finishes the
# next backbone of the chunk is started (no static striding, so a slow backbone no
# longer leaves the other workers idle). Needs bash >= 4.3 (wait -n).
TOTAL_B=$(( END - START + 1 ))
FAIL=0

if [ "${WORKERS}" -eq 1 ]; then
    K=0
    for IDX in $(seq "${START}" "${END}"); do
        K=$(( K + 1 ))
        process_backbone "${IDX}" "${K}/${TOTAL_B}"
    done
else
    if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 3 ]; }; then
        echo "[ERROR] bash >= 4.3 is required for the dynamic queue (have ${BASH_VERSION})"
        exit 1
    fi
    # One log file per backbone (live progress: tail -f); appended to this log in
    # backbone order when the task is done.
    BLOG_PREFIX="${OUT}/logs/mpnn_r${ROUND}_task${SLURM_ARRAY_TASK_ID}_b"
    echo "dynamic queue: ${TOTAL_B} backbone(s), ${WORKERS} in flight; live logs: ${BLOG_PREFIX}*.log"
    RUNNING=0
    K=0
    for IDX in $(seq "${START}" "${END}"); do
        K=$(( K + 1 ))
        while [ "${RUNNING}" -ge "${WORKERS}" ]; do
            wait -n || FAIL=1
            RUNNING=$(( RUNNING - 1 ))
        done
        process_backbone "${IDX}" "${K}/${TOTAL_B}" > "${BLOG_PREFIX}${IDX}.log" 2>&1 &
        RUNNING=$(( RUNNING + 1 ))
    done
    while [ "${RUNNING}" -gt 0 ]; do
        wait -n || FAIL=1
        RUNNING=$(( RUNNING - 1 ))
    done
    for IDX in $(seq "${START}" "${END}"); do
        echo "------ backbone index ${IDX} ------"
        cat "${BLOG_PREFIX}${IDX}.log" && rm -f "${BLOG_PREFIX}${IDX}.log"
    done
    if [ "${FAIL}" -ne 0 ]; then
        echo "[ERROR] at least one backbone process crashed"
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
WORKERS=__RELAX_CPUS__

TOTAL=$(wc -l < "${LIST}")
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
[ "${END}" -gt "${TOTAL}" ] && END=${TOTAL}

echo "=== FastRelax round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}, backbones ${START}-${END} on $(hostname) ==="

# Container image: optionally run from node-local scratch ($SLURM_TMPDIR) instead of
# mounting the shared-filesystem .sif for every launch (squashfuse + network reads
# dominated startup with many concurrent launches).
#   none    : use the .sif as given
#   copy    : copy the .sif to local scratch (still a squashfs mount, but local reads)
#   sandbox : unpack the .sif to a directory on local scratch (no squashfuse at all)
IMG="__SIF__"
IMG_MODE="__IMG_MODE__"
if [ "${IMG_MODE}" != "none" ]; then
    if [ -z "${SLURM_TMPDIR:-}" ] || [ ! -d "${SLURM_TMPDIR}" ]; then
        echo "[WARN] SLURM_TMPDIR not set, using the original image"
    else
        LOCAL_ROOT="${SLURM_TMPDIR}/img_${SLURM_JOB_ID}"
        mkdir -p "${LOCAL_ROOT}"
        T0=$(date +%s)
        if [ "${IMG_MODE}" = "copy" ]; then
            cp "__SIF__" "${LOCAL_ROOT}/image.sif" && IMG="${LOCAL_ROOT}/image.sif"
        else
            singularity build --sandbox "${LOCAL_ROOT}/sandbox" "__SIF__" > "${LOCAL_ROOT}/build.log" 2>&1 \
                && IMG="${LOCAL_ROOT}/sandbox"
        fi
        if [ "${IMG}" = "__SIF__" ]; then
            echo "[WARN] local image setup (${IMG_MODE}) failed, using the original image"
            [ -f "${LOCAL_ROOT}/build.log" ] && tail -5 "${LOCAL_ROOT}/build.log"
        fi
        echo "image: ${IMG} (mode ${IMG_MODE}, setup $(( $(date +%s) - T0 )) s)"
    fi
fi

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
    NJOBS=$(wc -l < "${JOBS_TSV}")
    W=${WORKERS}
    [ "${W}" -gt "${NJOBS}" ] && W=${NJOBS}
    echo "Relaxing ${NJOBS} backbone(s) with ${W} parallel worker(s)"

    # FastRelax is single-threaded: one thread per process, parallelism comes from
    # running independent worker processes on disjoint shards of the jobs list.
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

    PIDS=()
    for (( k=0; k<W; k++ )); do
        SHARD="${OUT}/logs/relax_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.shard${k}.tsv"
        SLOG="${OUT}/logs/relax_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.shard${k}.log"
        WD="/tmp/relax_${SLURM_JOB_ID}_${k}"      # private cwd per worker (no scratch-file clashes)
        mkdir -p "${WD}"
        # strided split so each shard gets a similar mix of backbones
        awk -v n="${W}" -v k="${k}" '(NR-1)%n==k' "${JOBS_TSV}" > "${SHARD}"
        singularity exec --pwd "${WD}" -B __BIND_PATH__ "${IMG}" \
            python3 "${OUT}/03b_fastrelax_worker.py" \
            --jobs-tsv "${SHARD}" \
            --xml "${OUT}/fast_relax_cyclize.xml" > "${SLOG}" 2>&1 &
        PIDS+=($!)
    done

    FAIL=0
    for k in "${!PIDS[@]}"; do
        wait "${PIDS[$k]}" || { echo "[WARN] relax worker ${k} exited non-zero"; FAIL=1; }
        echo "----- relax worker ${k} log -----"
        cat "${OUT}/logs/relax_r${ROUND}_task${SLURM_ARRAY_TASK_ID}.shard${k}.log"
        rm -rf "/tmp/relax_${SLURM_JOB_ID}_${k}"
    done
    exit ${FAIL}
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
    ap.add_argument("--mpnn-time", default="01:00:00")
    ap.add_argument("--mpnn-mem", default=None, help="Default: max(16, 4 x workers) GB")
    ap.add_argument("--mpnn-cpus", type=int, default=None, help="Default: max(2, workers)")
    ap.add_argument("--mpnn-workers", type=int, default=1,
                    help="Concurrent ProteinMPNN processes sharing the GPU inside each MPNN "
                         "array task (default 1 = previous behavior)")
    ap.add_argument("--mpnn-temps", default="0.0001 0.02 0.04 0.06 0.08 0.10 0.12 0.14",
                    help="Space-separated sampling temperatures, all sampled in ONE ProteinMPNN "
                         "call; lowest temperature with a valid sequence wins (default = the old "
                         "retry ladder)")
    ap.add_argument("--num-seq-per-temp", type=int, default=20,
                    help="Sequences sampled per temperature (default 20, as before per attempt)")
    ap.add_argument("--mpnn-batch-size", type=int, default=20,
                    help="ProteinMPNN --batch_size: samples decoded in parallel on the GPU. Must "
                         "divide --num-seq-per-temp. 1 = old behaviour (sequential decoding).")
    ap.add_argument("--local-image", choices=["none", "copy", "sandbox"], default="none",
                    help="Where containers run from: none = the .sif as given; copy = copy the "
                         ".sif to $SLURM_TMPDIR at the start of each task; sandbox = unpack it "
                         "there with 'singularity build --sandbox' (no squashfuse mounts). "
                         "Applies to MPNN and relax tasks.")
    ap.add_argument("--no-mps", action="store_true",
                    help="Do not start CUDA MPS when --mpnn-workers > 1")
    ap.add_argument("--relax-time", default="01:00:00")
    ap.add_argument("--relax-mem", default=None, help="Default: max(8, 4 x relax-cpus) GB")
    ap.add_argument("--relax-cpus", type=int, default=1,
                    help="Parallel FastRelax worker processes per relax array task (also sets "
                         "--cpus-per-task). Capped at the number of backbones in the chunk.")
    ap.add_argument("--submit", action="store_true", help="Run submit_all.sh after generating")
    args = ap.parse_args()

    if args.mpnn_workers < 1:
        sys.exit("--mpnn-workers must be >= 1")
    try:
        temps = [float(t) for t in args.mpnn_temps.split()]
    except ValueError:
        sys.exit("--mpnn-temps must be space-separated numbers")
    if not temps:
        sys.exit("--mpnn-temps is empty")
    if args.mpnn_batch_size < 1 or args.num_seq_per_temp % args.mpnn_batch_size:
        sys.exit("--mpnn-batch-size must be >= 1 and divide --num-seq-per-temp")
    mpnn_cpus = args.mpnn_cpus or max(2, args.mpnn_workers)
    mpnn_mem = args.mpnn_mem or f"{max(16, 4 * args.mpnn_workers)}G"
    if args.relax_cpus < 1:
        sys.exit("--relax-cpus must be >= 1")
    relax_mem = args.relax_mem or f"{max(8, 4 * args.relax_cpus)}G"
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
        "MPNN_TEMPS": " ".join(args.mpnn_temps.split()), "NUM_SEQ": args.num_seq_per_temp,
        "MPNN_BATCH": args.mpnn_batch_size, "IMG_MODE": args.local_image,
        "RELAX_TIME": args.relax_time, "RELAX_MEM": relax_mem, "RELAX_CPUS": args.relax_cpus,
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
    print(f"MPNN: one call/backbone, {len(temps)} temperature(s) x {args.num_seq_per_temp} "
          f"samples (batch {args.mpnn_batch_size})")
    print(f"Relax workers per task: {args.relax_cpus} (mem {relax_mem})")
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
