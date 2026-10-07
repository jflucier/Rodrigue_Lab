#!/usr/bin/env python3
"""
04_afcycdesign_filter.py

Predicts target-macrocycle complex binding modes with AfCycDesign (one SLURM
array task = one chunk of designs) and logs per-design metrics to a
task-private CSV:  <out-dir>/logs/afcyc_r<round>_task<task-id>.csv

Design notes
- Every row is appended and flushed as soon as its prediction finishes, so a
  timeout/OOM loses at most the design in flight.
- Failed predictions are logged with passed=ERROR (and the message) so they are
  not silently retried forever. To retry them, delete those rows from the CSV.
- Resume: designs already present in this task's CSV are skipped.
- `passed` = iPAE < cutoff AND Ca RMSD < cutoff AND pLDDT > cutoff.
  The SI filters on iPAE and RMSD only; pLDDT is always recorded, but the gate
  is disabled by default (--plddt-cutoff 0.0).
- iPAE is ColabDesign's i_pae, already on a 0-1 scale (the SI's "normalized
  iPAE"), so no extra division is applied. The gate uses the MEAN over the models
  run, like the ensemble-averaged RMSD. The SI example used model [0] only; that
  value is logged as ipae_m0 (and the worst model as ipae_max) for comparison.
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    import numpy as np
except ImportError:
    np = None

try:
    from colabdesign import mk_afdesign_model, clear_mem
except ImportError:
    mk_afdesign_model = None
    clear_mem = None

CSV_HEADERS = ["design", "rmsd", "ipae", "ipae_m0", "ipae_max", "plddt", "passed", "error"]


def add_cyclic_offset(self, offset_type=2):
    """Cyclic offset patch from the sokrypton/ColabDesign notebook."""

    def cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i, i + L], -1)
        offset = i[:, None] - i[None, :]
        c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))
        if offset_type == 1:
            c_offset = c_offset
        elif offset_type >= 2:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]
        if offset_type == 3:
            idx = np.abs(c_offset) > 2
            c_offset[idx] = (32 * c_offset[idx]) / abs(c_offset[idx])
        return c_offset * np.sign(offset)

    idx = self._inputs["residue_index"]
    offset = np.array(idx[:, None] - idx[None, :])

    if self.protocol == "binder":
        c_offset = cyclic_offset(self._binder_len)
        offset[self._target_len:, self._target_len:] = c_offset

    if self.protocol in ["fixbb", "partial", "hallucination"]:
        Ln = 0
        for L in self._lengths:
            offset[Ln:Ln + L, Ln:Ln + L] = cyclic_offset(L)
            Ln += L
    self._inputs["offset"] = offset


def predict_one(pdb_path, pred_dir, binder_chain, target_chain, max_binder_len):
    """Forward co-complex prediction. Returns (rmsd, ipae_mean, ipae_m0, ipae_max, plddt)."""
    clear_mem()
    model = mk_afdesign_model("binder", use_multimer=True, data_dir="/opt/ColabDesign")
    model.prep_inputs(
        str(pdb_path),
        binder_chain=binder_chain,
        target_chain=target_chain,
        use_binder_template=False,
        use_multimer=True,
        use_initial_guess=True,
        data_dir="/opt/ColabDesign"
    )
    # Guard against swapped chains: the macrocycle must be the short chain.
    blen, tlen = int(model._binder_len), int(model._target_len)
    if blen > max_binder_len or blen >= tlen:
        raise ValueError(
            f"chain assignment looks wrong: binder chain {binder_chain} has {blen} "
            f"residues, target chain {target_chain} has {tlen} "
            f"(max-binder-len={max_binder_len})")
    # print("after chain validation")
    add_cyclic_offset(model, offset_type=2)
    model.set_seq(mode="wildtype")
    model.set_opt(num_recycles=1)
    model.predict(
        models=["model_1_multimer_v3", "model_2_multimer_v3"],
        verbose=True
    )
    # print("after predict")
    model.save_pdb(str(pred_dir / f"{pdb_path.stem}_prediction.pdb"))
    # print("after save_pdb")
    rmsd = float(model.aux["losses"]["rmsd"])
    # i_pae per model; the gate uses the mean over the models run (consistent with
    # the ensemble-averaged RMSD). Model 0 alone is what the SI example script used.
    ipae_all = np.asarray(model.aux["all"]["losses"]["i_pae"], dtype=float).reshape(-1)
    ipae, ipae_m0, ipae_max = float(ipae_all.mean()), float(ipae_all[0]), float(ipae_all.max())
    # print("fetch stats")
    # Confidence is aux["plddt"] (per residue, target first then binder).
    # aux["losses"]["plddt"] is a LOSS (1 - mean pLDDT), so it is not used here.
    try:
        per_res = np.asarray(model.aux["plddt"], dtype=float).reshape(-1)
        plddt = float(per_res[-blen:].mean())
        if plddt > 1.0:
            plddt /= 100.0
    except Exception:
        plddt = float("nan")
    return rmsd, ipae, ipae_m0, ipae_max, plddt


def read_done(csv_path):
    done = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("design"):
                    done.add(row["design"])
    return done


def append_row(csv_path, row):
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
        if new_file:
            w.writeheader()
        w.writerow(row)
        fh.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-tsv", required=True,
                    help="TSV for this chunk; the LAST column is the relaxed complex PDB")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--ipae-cutoff", "--norm-ipae-cutoff", dest="ipae_cutoff", type=float, default=0.30,
                    help="ColabDesign i_pae is already on a 0-1 scale (no further normalization)")
    ap.add_argument("--rmsd-cutoff", type=float, default=1.5)
    ap.add_argument("--plddt-cutoff", type=float, default=0.0,
                    help="0 disables the pLDDT gate (pLDDT is still recorded)")
    ap.add_argument("--binder-chain", default="A")
    ap.add_argument("--target-chain", default="B")
    ap.add_argument("--max-binder-len", type=int, default=30)
    args = ap.parse_args()

    jobs = []
    with open(args.jobs_tsv) as fh:
        for line in fh:
            if line.strip():
                jobs.append(Path(line.rstrip("\n").split("\t")[-1]))
    if not jobs:
        print("Nothing to process in this chunk.")
        return

    log_dir = Path(args.out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"afcyc_r{args.round}_task{args.task_id}.csv"

    done = read_done(csv_path)
    todo = [p for p in jobs if p.stem not in done]
    print(f"[*] {len(jobs)} designs in chunk, {len(done)} already logged, {len(todo)} to run")

    if todo:
        if np is None or mk_afdesign_model is None:
            sys.exit("numpy and/or ColabDesign are not importable in this environment.")

        for k, pdb_path in enumerate(todo, 1):
            print(f"[{k}/{len(todo)}] {pdb_path.name}", flush=True)
            pred_dir = args.out_dir / "predictions"
            pred_dir.mkdir(exist_ok=True)
            row = {h: "" for h in CSV_HEADERS}
            row["design"] = pdb_path.stem
            try:
                rmsd, ipae, ipae_m0, ipae_max, plddt = predict_one(
                    pdb_path, pred_dir, args.binder_chain, args.target_chain,
                    args.max_binder_len)
                plddt_ok = (plddt > args.plddt_cutoff) if args.plddt_cutoff > 0 else True
                passed = (ipae < args.ipae_cutoff) and (rmsd < args.rmsd_cutoff) and plddt_ok
                print(f"  iPAE mean {ipae:.3f} (m0 {ipae_m0:.3f}, max {ipae_max:.3f}) RMSD {rmsd:.2f} A "
                      f"pLDDT {plddt:.2f} passed={passed}", flush=True)
                row.update(rmsd=f"{rmsd:.4f}", ipae=f"{ipae:.4f}", ipae_m0=f"{ipae_m0:.4f}", ipae_max=f"{ipae_max:.4f}",
                           plddt=f"{plddt:.4f}", passed=str(passed))
            except Exception as e:
                msg = " ".join(str(e).split())[:200]
                print(f"  [WARN] prediction failed: {msg}", flush=True)
                row.update(passed="ERROR", error=msg)
            append_row(csv_path, row)

    # Summary for this task's CSV
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    n_ok = [r for r in rows if r["passed"] in ("True", "False")]
    n_pass = sum(r["passed"] == "True" for r in n_ok)
    n_err = sum(r["passed"] == "ERROR" for r in rows)
    print("=" * 60)
    print(f"Task {args.task_id} round {args.round}: {len(rows)} rows, "
          f"{n_pass} passed / {len(n_ok)} predicted, {n_err} errors")
    print("=" * 60)


if __name__ == "__main__":
    main()
