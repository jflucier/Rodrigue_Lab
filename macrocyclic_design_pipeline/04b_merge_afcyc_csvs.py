#!/usr/bin/env python
"""
04c_merge_afcyc_csvs.py

Merges the per-array-task CSVs written by 04_afcycdesign_filter.py
(afcyc_r<round>_task<n>.csv, one per SLURM array task -- see
04_generate_afcyc_pipeline.py) into one summary CSV per round, de-duplicated
by design name (last write wins, matching the per-task resume behavior).

This replaces a single shared GLOBAL_CSV that every array task appended to
concurrently -- on NFS, concurrent multi-writer appends from up to
--max-concurrent tasks are not guaranteed atomic and risk interleaved or
corrupted rows. Each task now owns its own file; this script is the one
place results get combined, run once after the arrays finish.

Usage:
    python 04c_merge_afcyc_csvs.py <out-dir> --round 4 \
        --out-csv <out-dir>/afcyc_filter_summary_r4.csv

Feed the resulting file to 04b_derive_target_cutoff.py.
"""
import argparse
import csv
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="The --out-dir shared with 04_generate_afcyc_pipeline.py")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out-csv", default=None,
                     help="Default: <out_dir>/afcyc_filter_summary_r<round>.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pattern = f"afcyc_r{args.round}_task*.csv"
    parts = sorted(out_dir.glob(f"logs/{pattern}"))
    if not parts:
        sys.exit(f"No files matching logs/{pattern} under {out_dir}")

    rows_by_design = {}
    fieldnames = None
    for part in parts:
        with open(part, newline="") as fh:
            reader = csv.DictReader(fh)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                sys.exit(f"{part}: column mismatch ({reader.fieldnames} vs {fieldnames})")
            for row in reader:
                rows_by_design[row["design"]] = row  # last write wins on dupes

    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"afcyc_filter_summary_r{args.round}.csv"
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for design in sorted(rows_by_design):
            writer.writerow(rows_by_design[design])

    n_passed = sum(1 for r in rows_by_design.values() if r.get("passed", "").lower() == "true")
    print(f"Merged {len(parts)} task CSVs -> {len(rows_by_design)} designs "
          f"({n_passed} passed) -> {out_csv}")


if __name__ == "__main__":
    main()
