#!/usr/bin/env python
"""
04_afcycdesign_filter.py

Replicates Section 2.3 of Rettie et al. 2025 (Nature Chemical Biology).
Predicts the target-macrocycle co-complex binding modes using AfCycDesign
and strictly filters on three hard constraints:
  - Normalized iPAE (iPAE/32) < 0.30
  - C-alpha RMSD to design model < 1.5 A
  - Average backbone pLDDT > 0.80 (80.0)

Tracks total pass-rate percentages to match paper benchmarks (~10-20% pass).
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
    """Runs forward co-complex structure prediction and extracts iPAE, RMSD, and pLDDT."""
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

    # --- EXTRACT ALL METRICS ACCURATELY ---
    rmsd = float(model.aux["losses"]["rmsd"])
    ipae = float(model.aux["all"]["losses"]["i_pae"][0])

    # Extract average pLDDT (ColabDesign stores it as a 0-1 or 0-100 float depending on version)
    # We normalize it to a 0.0 - 1.0 scale to precisely match your target constraint (pLDDT > 0.8)
    raw_plddt = float(model.aux["losses"]["plddt"])
    plddt = raw_plddt / 100.0 if raw_plddt > 1.0 else raw_plddt

    # Append directly to legacy score.sc file (including plddt for data preservation)
    with open(score_sc_path, "a") as outfile:
        outfile.write(f"{pdb_path.stem},{ipae},{rmsd},{plddt:.4f}\n")

    return rmsd, ipae, plddt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs-dir", required=True, help="Directory containing input relaxed complex PDBs")
    ap.add_argument("--out-csv", default="afcyc_filter_summary.csv", help="Clean metadata summary file")
    ap.add_argument("--norm-ipae-cutoff", type=float, default=0.30,
                    help="Normalized iPAE threshold limit (default 0.30)")
    ap.add_argument("--rmsd-cutoff", type=float, default=1.5, help="C-alpha RMSD threshold limit")
    ap.add_argument("--plddt-cutoff", type=float, default=0.80, help="Minimum average backbone pLDDT threshold")
    ap.add_argument("--ipae-norm", type=float, default=32.0, help="iPAE normalization divisor")
    ap.add_argument("--dry-run", action="store_true", help="Count input structures and exit")
    args = ap.parse_args()

    designs_dir = Path(args.designs_dir)
    pdbs = sorted(designs_dir.rglob("*.pdb"))
    if not pdbs:
        sys.exit(f"No PDB structural designs found under {designs_dir}")

    if args.dry_run:
        print(f"Would evaluate {len(pdbs)} structures.")
        return

    if mk_afdesign_model is None:
        sys.exit("ColabDesign is not importable. Run this inside the built image container environment.")

    pred_dir = designs_dir.parent / "afcyc_predictions"
    pred_dir.mkdir(exist_ok=True)

    score_sc_path = designs_dir.parent / "score.sc"

    print(f"Evaluating {len(pdbs)} structures...")
    print(
        f"Constraints: Norm iPAE < {args.norm_ipae_cutoff} | RMSD < {args.rmsd_cutoff} A | pLDDT > {args.plddt - cutoff}")

    rows = []
    for k, pdb in enumerate(pdbs, 1):
        print(f"[{k}/{len(pdbs)}] Predicting binding mode: {pdb.name}", flush=True)
        try:
            rmsd, ipae, plddt = predict_one(pdb, pred_dir, score_sc_path)
            norm_ipae = ipae / args.ipae_norm

            # Evaluate all three constraints concurrently
            passed_ipae = norm_ipae < args.norm_ipae_cutoff
            passed_rmsd = rmsd < args.rmsd_cutoff
            passed_plddt = plddt > args.plddt_cutoff
            passed = passed_ipae and passed_rmsd and passed_plddt

            print(
                f"  --> iPAE: {ipae:.2f} (Norm: {norm_ipae:.2f}), RMSD: {rmsd:.2f}A, pLDDT: {plddt:.2f} [Passed={passed}]",
                flush=True)

            rows.append({
                "design": pdb.stem, "rmsd": rmsd, "ipae": ipae,
                "normalized_ipae": norm_ipae, "plddt": plddt, "passed": passed
            })
        except Exception as e:
            print(f"  --> [WARN] Prediction failed for {pdb.name}: {e}", flush=True)
            continue

    # Save summary data array
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["design", "rmsd", "ipae", "normalized_ipae", "plddt", "passed"])
        writer.writeheader()
        writer.writerows(rows)

    # Calculate real-time pass percentages for cluster tracking
    total_processed = len(rows) if len(rows) > 0 else 1
    n_passed = sum(r["passed"] for r in rows)
    pass_rate = (n_passed / total_processed) * 100

    print("\n" + "=" * 60)
    print("📈 PIPELINE SCREENING MATRIX CONVERGENCE")
    print("=" * 60)
    print(f"Total Designs Processed : {total_processed}")
    print(f"Total Designs Passed    : {n_passed}")
    print(f"Calculated Pass Rate    : {pass_rate:.2f}%")
    print(f"Expected Target Corridor: 10.00% - 20.00% (MDM2 Baseline Benchmarks)")
    print("=" * 60)
    print(f"Summary metrics saved cleanly to {args.out_csv}")


if __name__ == "__main__":
    main()
