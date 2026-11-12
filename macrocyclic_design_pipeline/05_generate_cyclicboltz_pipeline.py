#!/usr/bin/env python3
"""
05_generate_cyclicboltz_pipeline.py

Orthogonal structural validation of AfCycDesign hits with CyclicBoltz1.

Workflow:
  1. python 05_generate_cyclicboltz_pipeline.py --afcyc-out 04_afcycdesign_outputs/Mcl1 \
         --mpnn-relax-out <03 out dir> --out-dir 05_cyclicboltz_outputs/Mcl1 --sif ... [--submit]
  2. bash <out-dir>/submit_cboltz.sh
  3. python 05b_merge_cyclicboltz_csvs.py <out-dir> --expected-tasks T

Hits = rows with passed==True in <afcyc-out>/logs/afcyc_r*_task*.csv (all rounds).
The relaxed complex fed to AfCycDesign is <mpnn-relax-out>/<stem>/round<N>/<stem>_r<N>.pdb.
Consensus rule: Boltz-vs-AfCycDesign binder CA RMSD (target-fitted) < --rmsd-cutoff (2 A).
"""
import argparse
import csv
import glob
import math
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=mcy_cboltz
#SBATCH --partition=__QUEUE__
#SBATCH --output=__OUT__/logs/cboltz_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --requeue

set -uo pipefail
OUT="__OUT__"
LIST="${OUT}/hits.tsv"
CHUNK=__CHUNK__
W=__WORKERS__
TID=${SLURM_ARRAY_TASK_ID}

TOTAL=$(wc -l < "${LIST}")
START=$(( (TID - 1) * CHUNK + 1 ))
END=$(( TID * CHUNK ))
[ "${END}" -gt "${TOTAL}" ] && END=${TOTAL}
[ "${START}" -gt "${TOTAL}" ] && { echo "nothing to do"; exit 0; }

JOBS="${OUT}/logs/cboltz_task${TID}.tsv"
sed -n "${START},${END}p" "${LIST}" > "${JOBS}"
N_JOBS=$(wc -l < "${JOBS}")
TASK_CSV="${OUT}/logs/cboltz_task${TID}.csv"
echo "=== CyclicBoltz1 task ${TID}: entries ${START}-${END} (${N_JOBS}) on $(hostname), ${W} worker(s) ==="

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_worker () {  # $1 jobs tsv, $2 task id, $3 log
    singularity exec --nv -B __BIND_PATH__,__SCRIPT_DIR__ "__SIF__" \
        python3 "__SCRIPT_DIR__/05_cyclicboltz_validate.py" \
        --jobs-tsv "$1" --out-dir "${OUT}" --task-id "$2" \
        --binder-chain __BINDER_CHAIN__ --target-chain __TARGET_CHAIN__ \
        --rmsd-cutoff __RMSD_CUTOFF__ --batch-size __BOLTZ_BATCH__ \
        --recycling-steps __RECYCLING__ --sampling-steps __SAMPLING__ \
        --target-msa "__TARGET_MSA__" --boltz-cmd "__BOLTZ_CMD__" \
        __CACHE_ARG__ __PRED_DIR_ARG__ __EXTRA_ARG__ > "$3" 2>&1
}

if [ "${W}" -le 1 ] || [ "${N_JOBS}" -le 1 ]; then
    run_worker "${JOBS}" "${TID}" "${OUT}/logs/cboltz_task${TID}.worker.log"
    RC=$?
    cat "${OUT}/logs/cboltz_task${TID}.worker.log"
    [ "${RC}" -eq 0 ] && rm -f "${JOBS}" "${OUT}/logs/cboltz_task${TID}.worker.log"
    exit "${RC}"
fi

[ "${W}" -gt "${N_JOBS}" ] && W=${N_JOBS}
PIDS=(); SHARDS=()
# resume: shard CSVs from an earlier attempt are kept and the worker skips finished designs
[ -f "${TASK_CSV}" ] && echo "[*] existing task CSV found; merged below with new shard results"
for (( k=0; k<W; k++ )); do
    SID=$(( 100000 + TID * 100 + k ))
    PFX="${OUT}/logs/cboltz_task${TID}.shard${k}"
    awk -v n="${W}" -v k="${k}" '(NR-1)%n==k' "${JOBS}" > "${PFX}.tsv"
    SHARDS+=("${OUT}/logs/cboltz_task${SID}.csv")
    run_worker "${PFX}.tsv" "${SID}" "${PFX}.log" &
    PIDS+=($!)
    sleep 2
done

FAIL=0
for k in "${!PIDS[@]}"; do
    wait "${PIDS[$k]}" || { echo "[WARN] worker ${k} exited non-zero"; FAIL=1; }
    echo "----- worker ${k} log -----"; cat "${OUT}/logs/cboltz_task${TID}.shard${k}.log"
done

EXISTING=()
for f in "${SHARDS[@]}"; do [ -s "${f}" ] && EXISTING+=("${f}"); done
[ "${#EXISTING[@]}" -eq 0 ] && { echo "[ERROR] no shard CSV written"; exit 1; }
awk 'FNR==1 && NR!=1 {next} {print}' "${EXISTING[@]}" > "${TASK_CSV}.tmp" && mv "${TASK_CSV}.tmp" "${TASK_CSV}"
echo "[*] merged ${#EXISTING[@]} shard CSV(s) -> ${TASK_CSV} ($(( $(wc -l < "${TASK_CSV}") - 1 )) rows / ${N_JOBS} designs)"
if [ "${FAIL}" -eq 0 ]; then
    rm -f "${JOBS}"
    for (( k=0; k<W; k++ )); do rm -f "${OUT}/logs/cboltz_task${TID}.shard${k}."{tsv,log}; done
    rm -f "${EXISTING[@]}"
fi
exit "${FAIL}"
"""

SUBMIT = r"""#!/bin/bash
# submit_cboltz.sh [TASK_RANGE]   e.g. bash submit_cboltz.sh 1-3   (default: all tasks)
set -uo pipefail
RANGE="${1:-1-__N_TASKS__}"
sbatch --array=${RANGE}%__MAX_CONC__ "__OUT__/cboltz_step.slurm"
"""


def fill(t, m):
    for k, v in m.items():
        t = t.replace(f"__{k}__", str(v))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--afcyc-out", required=True, help="04 output dir (contains logs/afcyc_r*_task*.csv)")
    ap.add_argument("--mpnn-relax-out", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sif", required=True, help="CyclicBoltz1 container")
    ap.add_argument("--afcyc-pred-dir", default=None,
                    help="Dir with AfCycDesign predicted complexes (<design>*.pdb); "
                         "default: use the relaxed input complex as reference")
    ap.add_argument("--max-hits", type=int, default=None, help="cap (after sorting by iPAE); use 2-10 for a smoke test")
    ap.add_argument("--sort-by-ipae", action="store_true", default=True)
    ap.add_argument("--queue", default="gh-bio")
    ap.add_argument("--bind-path", default="/net/nfs-bio")
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--max-concurrent", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2, help="concurrent boltz processes per GPU task")
    ap.add_argument("--time", default="02:00:00")
    ap.add_argument("--mem", default=None, help="default 24G x workers")
    ap.add_argument("--cpus", type=int, default=None, help="default 4 x workers")
    ap.add_argument("--binder-chain", default="A")
    ap.add_argument("--target-chain", default="B")
    ap.add_argument("--rmsd-cutoff", type=float, default=2.0)
    ap.add_argument("--target-msa", default="empty", help="a3m/csv for the target or 'empty'")
    ap.add_argument("--boltz-cmd", default="boltz")
    ap.add_argument("--boltz-cache", default=None, help="Boltz weights/ccd cache dir (needed offline)")
    ap.add_argument("--boltz-batch", type=int, default=25, help="designs per `boltz predict` call")
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    afc, mpnn, out = Path(a.afcyc_out).resolve(), Path(a.mpnn_relax_out).resolve(), Path(a.out_dir).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)

    hits, n_rows = {}, 0
    for f in glob.glob(str(afc / "logs" / "afcyc_r*_task*.csv")):
        m = re.search(r"_task(\d+)\.csv$", f)
        if not m or int(m.group(1)) >= 100000:   # skip leftover worker shards
            continue
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                n_rows += 1
                if r.get("passed") == "True":
                    try:
                        hits[r["design"]] = float(r["ipae"])
                    except ValueError:
                        hits[r["design"]] = 9.9
    print(f"AfCycDesign rows read: {n_rows}; passing designs: {len(hits)}")

    ordered = sorted(hits, key=lambda d: hits[d])
    if a.max_hits:
        ordered = ordered[:a.max_hits]
    lines, missing = [], 0
    for d in ordered:
        m = re.match(r"(.+)_r(\d+)$", d)
        if not m:
            missing += 1
            continue
        p = mpnn / m.group(1) / f"round{m.group(2)}" / f"{d}.pdb"
        if p.is_file():
            lines.append(f"{d}\t{p}")
        else:
            missing += 1
    if missing:
        print(f"[WARN] {missing} hits skipped (relaxed PDB not found)")
    if not lines:
        sys.exit("No hits to validate.")
    (out / "hits.tsv").write_text("\n".join(lines) + "\n")

    n_tasks = math.ceil(len(lines) / a.chunk_size)
    mem = a.mem or f"{24 * a.workers}G"
    cpus = a.cpus or 4 * a.workers
    m = {
        "OUT": out, "QUEUE": a.queue, "CPUS": cpus, "MEM": mem, "TIME": a.time,
        "CHUNK": a.chunk_size, "WORKERS": a.workers, "BIND_PATH": a.bind_path,
        "SCRIPT_DIR": SCRIPT_DIR, "SIF": Path(a.sif).resolve(),
        "BINDER_CHAIN": a.binder_chain, "TARGET_CHAIN": a.target_chain,
        "RMSD_CUTOFF": a.rmsd_cutoff, "BOLTZ_BATCH": a.boltz_batch,
        "RECYCLING": a.recycling_steps, "SAMPLING": a.sampling_steps,
        "TARGET_MSA": a.target_msa, "BOLTZ_CMD": a.boltz_cmd,
        "CACHE_ARG": f'--boltz-cache "{a.boltz_cache}"' if a.boltz_cache else "",
        "PRED_DIR_ARG": f'--afcyc-pred-dir "{Path(a.afcyc_pred_dir).resolve()}"' if a.afcyc_pred_dir else "",
        "EXTRA_ARG": f'--extra-args "{a.extra_args}"' if a.extra_args else "",
        "N_TASKS": n_tasks, "MAX_CONC": a.max_concurrent,
    }
    (out / "cboltz_step.slurm").write_text(fill(TEMPLATE, m))
    sub = out / "submit_cboltz.sh"
    sub.write_text(fill(SUBMIT, m))
    sub.chmod(0o755)

    print("=" * 70)
    print(f"Designs to validate : {len(lines)}  ->  {n_tasks} array tasks (chunk {a.chunk_size})")
    print(f"Workers/task        : {a.workers} (cpus {cpus}, mem {mem}, time {a.time})")
    print(f"Reference pose      : {'AfCycDesign prediction dir' if a.afcyc_pred_dir else 'relaxed input complex'}")
    print(f"Consensus           : binder CA RMSD (target-fitted) < {a.rmsd_cutoff} A")
    print(f"Launch              : bash {sub}      (smoke test: bash {sub} 1-1)")
    print(f"Then                : python {SCRIPT_DIR}/05b_merge_cyclicboltz_csvs.py {out} --expected-tasks {n_tasks}")
    print("=" * 70)
    if a.submit:
        subprocess.run(["bash", str(sub)], check=True)


if __name__ == "__main__":
    main()
