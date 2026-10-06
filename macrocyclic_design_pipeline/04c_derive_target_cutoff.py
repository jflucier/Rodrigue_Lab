#!/usr/bin/env python3
"""
04c_derive_target_cutoff.py

Descriptive statistics of normalized iPAE for one target, to help choose a
per-target cutoff.

Context (Rettie et al., SI section 2.3): the authors chose normalized-iPAE
cutoffs "based on the iPAE distribution per target" -- 0.2 (MCL1), 0.3 (MDM2),
0.13 (GABARAP), 0.4 (RbtA) -- and noted iPAE > 0.4 was often a non-optimal
interface or a non-interacting peptide. No percentile rule is given, so
nothing below is "the paper's method". iPAE is a confidence metric, not an
affinity; iPAE filtering is only the first stage before Rosetta ddG/SAP/CMS
(and RF2 for some targets).

Usage:
    python 04c_derive_target_cutoff.py <out-dir>/afcyc_filter_summary_r4.csv \
        --max-rmsd 1.5 --budget 200
"""
import argparse
import sys

import numpy as np
import pandas as pd

PAPER_CUTOFFS = [0.13, 0.20, 0.30, 0.40]
CEILING = 0.40


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Merged summary CSV from 04b_merge_afcyc_csvs.py")
    ap.add_argument("--max-rmsd", type=float, default=1.5,
                    help="Ca RMSD cutoff in A (SI uses 1.5 for RbtA; keep equal to the filter script)")
    ap.add_argument("--ipae-norm", type=float, default=32.0,
                    help="Divisor used if only raw 'ipae' is present")
    ap.add_argument("--budget", type=int, default=200,
                    help="Number of designs you can afford to take to the Rosetta stage")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        sys.exit(f"Could not read {args.csv}: {e}")
    n_total = len(df)

    if "passed" in df.columns:
        df = df[df["passed"].astype(str).str.upper() != "ERROR"]
    if "normalized_ipae" not in df.columns and "ipae" in df.columns:
        df["normalized_ipae"] = pd.to_numeric(df["ipae"], errors="coerce") / args.ipae_norm
    missing = {"rmsd", "normalized_ipae"} - set(df.columns)
    if missing:
        sys.exit(f"Missing columns: {sorted(missing)} (have {list(df.columns)})")

    for c in ("rmsd", "normalized_ipae"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["rmsd", "normalized_ipae"])
    n_pred = len(df)
    pool = df[df["rmsd"] < args.max_rmsd]
    if pool.empty:
        sys.exit(f"No designs with RMSD < {args.max_rmsd} A out of {n_pred} predicted.")
    x = pool["normalized_ipae"].to_numpy()

    print(f"Rows in CSV: {n_total} | predicted OK: {n_pred} | RMSD < {args.max_rmsd} A: {len(pool)} "
          f"({100 * len(pool) / n_pred:.1f}%)")
    print("\nNormalized iPAE of the RMSD-passing pool (raw = x{:.0f}):".format(args.ipae_norm))
    for label, q in [("min", 0), ("p1", 1), ("p5", 5), ("p10", 10), ("p25", 25), ("median", 50)]:
        v = np.percentile(x, q)
        print(f"  {label:>6}: {v:.3f}  (raw {v * args.ipae_norm:.1f})")

    print("\nDesigns passing at candidate cutoffs (paper values):")
    print(f"  {'cutoff':>8} {'n pass':>8} {'% of RMSD-pool':>15} {'% of all predicted':>20}")
    for c in PAPER_CUTOFFS:
        n = int((x < c).sum())
        print(f"  {c:>8.2f} {n:>8d} {100 * n / len(pool):>14.2f}% {100 * n / n_pred:>19.2f}%")

    # Heuristic (not from the paper): loosest cutoff <= ceiling that fits the budget.
    grid = np.round(np.arange(0.05, CEILING + 1e-9, 0.01), 2)
    fits = [c for c in grid if (x < c).sum() <= args.budget]
    print(f"\nBudget heuristic (<= {args.budget} designs, ceiling {CEILING:.2f}):")
    if not fits:
        print(f"  Even iPAE < {grid[0]:.2f} passes {(x < grid[0]).sum()} designs; "
              "tighten with Rosetta metrics or raise the budget.")
    elif fits[-1] >= CEILING:
        n_all = int((x < CEILING).sum())
        print(f"  All {n_all} designs below the {CEILING:.2f} ceiling fit the budget.")
    else:
        c = fits[-1]
        print(f"  iPAE < {c:.2f} -> {(x < c).sum()} designs")

    n_below = int((x < CEILING).sum())
    if n_below < 20:
        print(f"\n[WARN] Only {n_below} designs have iPAE < {CEILING:.2f}; the pool is thin "
              "and any cutoff here is weakly supported.")
    print("\nNext step in the paper's workflow: Rosetta ddG / SAP / contact molecular surface "
          "on the survivors; the authors merged an iPAE-ranked and a Rosetta-ranked list.")


if __name__ == "__main__":
    main()
