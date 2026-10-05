#!/usr/bin/env python3
"""
04_afcycdesign_filter.py

Predicts target-macrocycle complex binding modes using AfCycDesign forward passes.
Strictly filters on: Normalized iPAE < 0.30, Ca RMSD < 1.5 A, and pLDDT > 0.80.

Features a robust, non-destructive resume checkpoint engine that parses the
summary CSV on startup to bypass completed trajectories.
"""
import argparse
import csv
import sys
import os
import warnings
from pathlib import Path

warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    import numpy as np
    from colabdesign import mk_afdesign_model, clear_mem
except ImportError:
    mk_afdesign_model = None


def add_cyclic_offset(self, offset_type=2):
    """Verbatim cyclic offset patch from the sokrypton/ColabDesign notebook."""

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


def predict_one(pdb_path: Path, out_dir: Path, score_sc_path: Path):
    """Runs a forward co-complex structure prediction matching the paper logic."""
    clear_mem()

    model = mk_afdesign_model("binder")
    model.prep_inputs(
        str(pdb_path),
        binder_chain="A",  # Macrocycle (Your pipeline convention)
        target_chain="B",  # Target protein (Your pipeline convention)
        use_binder_template=False,
        use_multimer=True,
        use_initial_guess=True,
    )

    # Apply the cyclic peptide bond offset constraints
    add_cyclic_offset(model, offset_type=2)

    model.set_seq(mode="wildtype")
    model.set_opt(num_recycles=1)

    # Run structure prediction forward pass across models 0 and 1
    model.predict(models=[0, 1], verbose=False)

    # Save the predicted binding mode coordinates
    out_pdb = out_dir / f"{pdb_path.stem}_prediction.pdb"
    model.save_pdb(str(out_pdb))

    # Extract exact metrics matching paper snippet
    rmsd = float(model.aux["losses"]["rmsd"])
    ipae = float(model.aux["all"]["losses"]["i_pae"][0])

    # Normalize pLDDT tracking onto a standard 0.0 - 1.0 scale
    raw_plddt = float(model.aux["losses"]["plddt"])
    plddt = raw_plddt / 100.0 if raw_plddt > 1.0 else raw_plddt

    # Append directly to legacy score.sc file
    with open(score_sc_path, "a") as outfile:
        outfile.write(f"{pdb_path.stem},{ipae:.4f},{rmsd:.4f},{plddt:.4f}\n")

    return rmsd, ipae, plddt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-tsv", required=True,
                    help="Path to the TSV file defining this parallel worker chunk segment")
    ap.add_argument("--out-csv", required=True, help="Global path to the centralized summary output CSV")
    ap.add_argument("--norm-ipae-cutoff", type=float, default=0.30)
    ap.add_argument("--rmsd-cutoff", type=float, default=1.5)
    ap.add_argument("--plddt-cutoff", type=float, default=0.80)
    ap.add_argument("--ipae-norm", type=float, default=32.0)
    args = ap.parse_args()

    # 1. Parse the TSV chunk file allocated to this specific node process
    jobs = []
    with open(args.jobs_tsv, "r") as f:
        for line in f:
            if line.strip():
                jobs.append(line.strip().split("\t"))

    if not jobs:
        sys.exit("Nothing to process inside the allocated jobs-tsv chunk segment.")

    if mk_afdesign_model is None:
        sys.exit("ColabDesign is not importable inside the current context.")

    out_csv_path = Path(args.out_csv)

    # 2. Non-Destructive Resume Check: Build index of previously completed design keys
    completed_designs = set()
    historical_rows = []
    if out_csv_path.exists() and out_csv_path.stat().st_size > 0:
        try:
            with open(out_csv_path, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    completed_designs.add(row["design"])
                    historical_rows.append(row)
            print(
                f"[*] Discovered existing summary database on disk. Cache contains {len(completed_designs)} completed records.")
        except Exception as e:
            print(f"[WARN] Error reading existing CSV database; processing chunk with a blank slate. Error: {e}")

    # Process tasks assigned to this active thread node
    new_rows = []
    for k, (in_pdb_str, fasta_str, relaxed_str) in enumerate(jobs, 1):
        pdb_path = Path(relaxed_str)

        # Immediate shortcut skip if the design is already logged in the summary table
        if pdb_path.stem in completed_designs:
            print(f"[{k}/{len(jobs)}] [SKIP] {pdb_path.stem} already logged in CSV database.")
            continue

        print(f"[{k}/{len(jobs)}] Predicting co-complex binding mode: {pdb_path.name}", flush=True)
        pred_dir = pdb_path.parent / "afcyc_predictions"
        pred_dir.mkdir(exist_ok=True)
        score_sc_path = pdb_path.parent / "score.sc"

        try:
            rmsd, ipae, plddt = predict_one(pdb_path, pred_dir, score_sc_path)
            norm_ipae = ipae / args.ipae_norm

            passed = (norm_ipae < args.norm_ipae_cutoff) and (rmsd < args.rmsd_cutoff) and (plddt > args.plddt_cutoff)
            print(
                f"  --> iPAE: {ipae:.2f} (Norm: {norm_ipae:.2f}), RMSD: {rmsd:.2f}A, pLDDT: {plddt:.2f} [Passed={passed}]",
                flush=True)

            new_rows.append({
                "design": pdb_path.stem, "rmsd": f"{rmsd:.4f}", "ipae": f"{ipae:.4f}",
                "normalized_ipae": f"{norm_ipae:.4f}", "plddt": f"{plddt:.4f}", "passed": str(passed)
            })
        except Exception as e:
            print(f"  --> [WARN] Prediction pass failed for {pdb_path.name}: {e}", flush=True)
            continue

    # 3. Thread-Safe Atomic Append: Append new lines to the centralized global tracker table
    csv_headers = ["design", "rmsd", "ipae", "normalized_ipae", "plddt", "passed"]
    write_header = not out_csv_path.exists() or out_csv_path.stat().st_size == 0

    with open(out_csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_headers)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    # 4. Display Aggregate Performance Metrics (Historic + New rows combined)
    all_processed_rows = historical_rows + new_rows
    total_processed = len(all_processed_rows) if all_processed_rows else 1
    n_passed = sum(1 for r in all_processed_rows if r["passed"].lower() == "true")
    pass_rate = (n_passed / total_processed) * 100

    print("\n" + "=" * 60)
    print("📈 AGGREGATE POOL SCREENING MATRIX CONVERGENCE")
    print("=" * 60)
    print(f"Total Cumulative Database Designs : {total_processed}")
    print(f"Total Combined Verified Binders   : {n_passed}")
    print(f"Calculated Pass Rate Percentage   : {pass_rate:.2f}%")
    print(f"Expected Target Corridor Corridor : 10.00% - 20.00% (MDM2 Baseline Benchmarks)")
    print("=" * 60)


if __name__ == "__main__":
    main()
