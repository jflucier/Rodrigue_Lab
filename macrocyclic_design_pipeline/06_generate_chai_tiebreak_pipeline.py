#!/usr/bin/env python3
"""
06_generate_chai_tiebreak_pipeline.py

Chai-1 tiebreaker for designs where AfCycDesign and CyclicBoltz1 disagree on the
binding pose (tiebreak_candidates.csv from 05b).

Workflow:
  1. python 06_generate_chai_tiebreak_pipeline.py --cboltz-out <05 out> --out-dir <06 out> \
         --chai-sif ... [--max-candidates 10 for a smoke test] [--submit]
  2. bash <out-dir>/submit_chai.sh [TASK_RANGE]
  3. python 06b_merge_chai_csvs.py <out-dir> --cboltz-out <05 out> --expected-tasks T

Per candidate, Chai-1 folds target + binder (5 -> --num-samples samples, ranked by Chai's
aggregate score). The top-ranked sample's binder CA RMSD (target-fitted) is compared with
 (a) the AfCycDesign pose and (b) the CyclicBoltz1 pose. Verdicts:
   chai_agrees_afcyc / chai_agrees_boltz / chai_agrees_both / no_consensus
Two of three models on one pose = rescued (majority vote).

NOTE: Chai-1 has no cyclic-offset support: the binder is folded as a linear peptide, so
its vote on the pose is weaker than CyclicBoltz1's. The target MSA is built once with
the ColabFold server (prep job); binders run with a query-only MSA.
"""
import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}

CHAI_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=mcy_chai
#SBATCH --partition=__QUEUE__
#SBATCH --output=__OUT__/logs/chai_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --requeue

set -uo pipefail
OUT="__OUT__"
LIST="${OUT}/candidates.tsv"
CHUNK=__CHUNK__
W=__WORKERS__
TID=${SLURM_ARRAY_TASK_ID}
NS=__NUM_SAMPLES__
TARGET="__TARGET_SEQ__"
MSA="${OUT}/msa/dir"
PRED="${OUT}/chai_predictions"
FASTA_DIR="${OUT}/fasta"
USE_MPS=__USE_MPS__
MPS_BIND=""
mkdir -p "${PRED}" "${FASTA_DIR}"

TOTAL=$(wc -l < "${LIST}")
START=$(( (TID - 1) * CHUNK + 1 ))
END=$(( TID * CHUNK ))
[ "${END}" -gt "${TOTAL}" ] && END=${TOTAL}
[ "${START}" -gt "${TOTAL}" ] && { echo "nothing to do"; exit 0; }

JOBS="${OUT}/logs/chai_task${TID}.tsv"
sed -n "${START},${END}p" "${LIST}" > "${JOBS}"
N_JOBS=$(wc -l < "${JOBS}")
[ "${W}" -gt "${N_JOBS}" ] && W=${N_JOBS}
echo "=== Chai-1 task ${TID}: entries ${START}-${END} (${N_JOBS}) on $(hostname), ${W} worker(s) ==="
: > "${OUT}/logs/chai_task${TID}.failed"

run_design () {   # $1 design  $2 binder sequence
    local D="$1" SEQ="$2"
    [ -s "${PRED}/${D}/scores.model_idx_$((NS-1)).npz" ] && return 0   # resume
    printf '>protein|name=binder\n%s\n>protein|name=target\n%s\n' "${SEQ}" "${TARGET}" > "${FASTA_DIR}/${D}.fasta"
    rm -rf "${PRED}/${D}.tmp" "${PRED}/${D}"
    if apptainer run --nv -B "__CHAI_MODELS__" -B "__BIND_PATH__" ${TMPDIR:+-B "${TMPDIR}"} ${MPS_BIND} \
            "__CHAI_SIF__" fold "${FASTA_DIR}/${D}.fasta" "${PRED}/${D}.tmp" \
            --msa-directory "${MSA}" \
            --num-diffn-samples ${NS} --num-trunk-recycles __RECYCLES__ \
            --num-diffn-timesteps __STEPS__ --seed __SEED__ __ESM_FLAG__ __EXTRA_ARGS__ < /dev/null \
        && [ -s "${PRED}/${D}.tmp/scores.model_idx_$((NS-1)).npz" ]; then
        mv "${PRED}/${D}.tmp" "${PRED}/${D}"
        rm -f "${FASTA_DIR}/${D}.fasta"
    else
        echo "[WARN] chai failed for ${D}"
        echo "${D}" >> "${OUT}/logs/chai_task${TID}.failed"
        rm -rf "${PRED}/${D}.tmp"
    fi
}

worker () {
    while IFS=$'\t' read -r D SEQ PDB; do
        run_design "${D}" "${SEQ}"
    done < "$1"
}

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

if [ "${W}" -le 1 ]; then
    worker "${JOBS}"
else
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
    for (( k=0; k<W; k++ )); do
        awk -v n="${W}" -v k="${k}" '(NR-1)%n==k' "${JOBS}" > "${OUT}/logs/chai_task${TID}.shard${k}.tsv"
        worker "${OUT}/logs/chai_task${TID}.shard${k}.tsv" > "${OUT}/logs/chai_task${TID}.shard${k}.log" 2>&1 &
        PIDS+=($!)
        sleep 5
    done
    for k in "${!PIDS[@]}"; do
        wait "${PIDS[$k]}" || echo "[WARN] worker ${k} exited non-zero"
        echo "----- worker ${k} log (tail) -----"; tail -n 40 "${OUT}/logs/chai_task${TID}.shard${k}.log"
    done
    [ "${USE_MPS}" -eq 1 ] && { echo quit | nvidia-cuda-mps-control || true; }
    rm -f "${OUT}/logs/chai_task${TID}.shard"*.tsv
fi

# Score (CPU, runs in the Chai container, which has numpy): Chai pose vs AfCycDesign and Boltz poses
apptainer exec -B "__BIND_PATH__,__SCRIPT_DIR__" "__CHAI_SIF__" __CHAI_PYTHON__ \
    "__SCRIPT_DIR__/06_chai_score.py" \
    --jobs-tsv "${JOBS}" --chai-dir "${PRED}" --num-samples ${NS} \
    --boltz-pred-dir "__BOLTZ_PRED_DIR__" __AFCYC_ARG__ \
    --rmsd-cutoff __RMSD_CUTOFF__ --out-csv "${OUT}/logs/chai_task${TID}.csv"
RC=$?
NFAIL=$(wc -l < "${OUT}/logs/chai_task${TID}.failed")
echo "[*] task ${TID}: scoring exit ${RC}; ${NFAIL} Chai failure(s) listed in logs/chai_task${TID}.failed"
[ "${RC}" -eq 0 ] && rm -f "${JOBS}"
exit "${RC}"
"""

PREP_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=mcy_chai_msa
#SBATCH --partition=__QUEUE__
#SBATCH --output=__OUT__/logs/chai_msa_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=__PREP_TIME__

# One-off: fold the first candidate with --use-msa-server so Chai writes the target MSA, then
# build the msa directory (target MSA + query-only MSA for every candidate binder).
set -uo pipefail
OUT="__OUT__"
PREP="${OUT}/msa/prep_out"
rm -rf "${PREP}" "${OUT}/msa/dir"
apptainer run --nv -B "__CHAI_MODELS__" -B "__BIND_PATH__" ${TMPDIR:+-B "${TMPDIR}"} \
    "__CHAI_SIF__" fold "${OUT}/msa/prep.fasta" "${PREP}" \
    --use-msa-server --num-diffn-samples 1 --num-trunk-recycles 1 --num-diffn-timesteps 50 \
    __ESM_FLAG__ < /dev/null || { echo "[ERROR] prep fold failed"; exit 1; }
apptainer exec -B "__BIND_PATH__,__SCRIPT_DIR__" "__CHAI_SIF__" __CHAI_PYTHON__ \
    "__SCRIPT_DIR__/06_prep_chai_msas.py" --prep-dir "${PREP}" --candidates "${OUT}/candidates.tsv" \
    --target-seq "__TARGET_SEQ__" --binder-seq "__PREP_BINDER__" --out-dir "${OUT}/msa/dir" \
    || { echo "[ERROR] building the msa directory failed"; exit 1; }
[ -f "${OUT}/msa/dir/READY" ] && echo "[*] chai MSA directory ready"
"""

SUBMIT_TEMPLATE = r"""#!/bin/bash
# submit_chai.sh [TASK_RANGE]   e.g. bash submit_chai.sh 1-1   (default: all tasks)
set -uo pipefail
RANGE="${1:-1-__N_TASKS__}"
DEP=""
if [ ! -f "__OUT__/msa/dir/READY" ]; then
    PREP=$(sbatch --parsable "__OUT__/prep_chai_msa.slurm") || exit 1
    PREP="${PREP%%;*}"
    echo "Submitted Chai MSA prep job ${PREP}"
    DEP="--dependency=afterok:${PREP}"
else
    echo "Using existing Chai MSA directory: __OUT__/msa/dir"
fi
sbatch ${DEP} --array=${RANGE}%__MAX_CONC__ "__OUT__/chai_step.slurm"
"""


def chain_seq(pdb, chain):
    seq = []
    for l in open(pdb):
        if l.startswith(("ATOM", "HETATM")) and l[12:16].strip() == "CA" and l[21] == chain \
                and l[16] in (" ", "A"):
            seq.append(AA3.get(l[17:20].strip(), "X"))
    return "".join(seq)


def fill(t, m):
    for k, v in m.items():
        t = t.replace(f"__{k}__", str(v))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cboltz-out", required=True,
                    help="05 output dir (tiebreak_candidates.csv, hits.tsv, boltz_predictions/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chai-sif", required=True)
    ap.add_argument("--chai-models", default="/home/jflucier/programs/Rodrigue_Lab/chai-1_models")
    ap.add_argument("--chai-python", default="python",
                    help="python inside the Chai container (MSA prep and scoring; needs numpy, pandas, pyarrow)")
    ap.add_argument("--afcyc-pred-dir", default=None, help="AfCycDesign predicted complexes "
                    "(default: relaxed input complex as AfCycDesign reference)")
    ap.add_argument("--binder-chain", default="A", help="binder chain in the relaxed input PDBs")
    ap.add_argument("--target-chain", default="B", help="target chain in the relaxed input PDBs")
    ap.add_argument("--max-candidates", type=int, default=None, help="cap, e.g. 10 for a smoke test")
    ap.add_argument("--queue", default="gh-bio")
    ap.add_argument("--bind-path", default="/net/nfs-bio")
    ap.add_argument("--chunk-size", type=int, default=60)
    ap.add_argument("--max-concurrent", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2, help="concurrent Chai processes per GPU task")
    ap.add_argument("--time", default="03:00:00")
    ap.add_argument("--prep-time", default="00:45:00")
    ap.add_argument("--mem", default=None, help="default 48G x workers")
    ap.add_argument("--cpus", type=int, default=None, help="default 4 x workers")
    ap.add_argument("--num-samples", type=int, default=3, help="Chai diffusion samples per design")
    ap.add_argument("--trunk-recycles", type=int, default=3)
    ap.add_argument("--diffn-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-esm", action="store_true", help="--no-use-esm-embeddings")
    ap.add_argument("--no-mps", action="store_true")
    ap.add_argument("--rmsd-cutoff", type=float, default=2.0)
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    cb, out = Path(a.cboltz_out).resolve(), Path(a.out_dir).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "msa").mkdir(exist_ok=True)

    paths = {}
    for l in (cb / "hits.tsv").read_text().splitlines():
        if l.strip():
            d, p = l.split("\t")[:2]
            paths[d] = p
    cands, no_path = [], 0
    with open(cb / "tiebreak_candidates.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("error"):
                continue
            d = r["design"]
            if d not in paths:
                no_path += 1
                continue
            cands.append((d, r["sequence"], paths[d]))
    if no_path:
        print(f"[WARN] {no_path} candidates without a relaxed PDB in hits.tsv; skipped")
    if a.max_candidates:
        cands = cands[:a.max_candidates]
    if not cands:
        sys.exit("No tiebreak candidates.")
    (out / "candidates.tsv").write_text("".join(f"{d}\t{s}\t{p}\n" for d, s, p in cands))

    tseq = chain_seq(cands[0][2], a.target_chain)
    if not tseq or "X" in tseq:
        sys.exit(f"Could not read target chain {a.target_chain} from {cands[0][2]}")
    prep_binder = cands[0][1]
    (out / "msa" / "prep.fasta").write_text(
        f">protein|name=binder\n{prep_binder}\n>protein|name=target\n{tseq}\n")

    n_tasks = math.ceil(len(cands) / a.chunk_size)
    use_mps = 1 if (a.workers > 1 and not a.no_mps) else 0
    mem = a.mem or f"{40 * a.workers}G"
    cpus = a.cpus or 4 * a.workers
    esm_flag = "--no-use-esm-embeddings" if a.no_esm else ""
    afcyc_pred = Path(a.afcyc_pred_dir).resolve() if a.afcyc_pred_dir else None
    m = {
        "OUT": out, "QUEUE": a.queue, "CPUS": cpus, "MEM": mem, "TIME": a.time, "PREP_TIME": a.prep_time,
        "CHUNK": a.chunk_size, "WORKERS": a.workers, "NUM_SAMPLES": a.num_samples,
        "TARGET_SEQ": tseq, "PREP_BINDER": prep_binder, "USE_MPS": use_mps,
        "CHAI_SIF": Path(a.chai_sif).resolve(),
        "CHAI_MODELS": a.chai_models, "CHAI_PYTHON": a.chai_python, "BIND_PATH": a.bind_path,
        "SCRIPT_DIR": SCRIPT_DIR, "RECYCLES": a.trunk_recycles, "STEPS": a.diffn_steps, "SEED": a.seed,
        "ESM_FLAG": esm_flag, "EXTRA_ARGS": a.extra_args,
        "BOLTZ_PRED_DIR": cb / "boltz_predictions",
        "AFCYC_ARG": f'--afcyc-pred-dir "{afcyc_pred}"' if afcyc_pred else "",
        "RMSD_CUTOFF": a.rmsd_cutoff, "N_TASKS": n_tasks, "MAX_CONC": a.max_concurrent,
    }
    (out / "chai_step.slurm").write_text(fill(CHAI_TEMPLATE, m))
    (out / "prep_chai_msa.slurm").write_text(fill(PREP_TEMPLATE, m))
    sub = out / "submit_chai.sh"
    sub.write_text(fill(SUBMIT_TEMPLATE, m))
    sub.chmod(0o755)

    print("=" * 70)
    print(f"Tiebreak candidates : {len(cands)}  ->  {n_tasks} array tasks (chunk {a.chunk_size})")
    print(f"Workers/task        : {a.workers} (cpus {cpus}, mem {mem}, MPS {'on' if use_mps else 'off'}, time {a.time})")
    print(f"Chai settings       : {a.num_samples} samples, {a.trunk_recycles} recycles, "
          f"{a.diffn_steps} steps, ESM {'off' if a.no_esm else 'on'}")
    print(f"AfCycDesign ref     : {afcyc_pred if afcyc_pred else 'relaxed input complex'}")
    print(f"Boltz poses         : {cb / 'boltz_predictions'}")
    print(f"Target length       : {len(tseq)}")
    print(f"Launch              : bash {sub}      (smoke test: bash {sub} 1-1 with --max-candidates 10)")
    print(f"Then                : python {SCRIPT_DIR}/06b_merge_chai_csvs.py {out} --cboltz-out {cb} --expected-tasks {n_tasks}")
    print("=" * 70)
    if a.submit:
        subprocess.run(["bash", str(sub)], check=True)


if __name__ == "__main__":
    main()
