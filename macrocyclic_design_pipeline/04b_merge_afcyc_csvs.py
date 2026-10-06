#!/usr/bin/env python3
"""
04b_merge_afcyc_csvs.py

Merges the per-array-task CSVs written by 04_afcycdesign_filter.py
(<out-dir>/logs/afcyc_r<round>_task<n>.csv) into one summary CSV per round,
de-duplicated by design name.

Duplicates: files are read oldest-first by modification time (rows within a
file in order), so the most recently written row wins.

Usage:
    python 04b_merge_afcyc_csvs.py <out-dir> --round 4 --expected-tasks 120

Feed the resulting file to 04c_derive_target_cutoff.py.
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="The --out-dir used with 04_generate_afcyc_pipeline.py")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out-csv", default=None,
                    help="Default: <out_dir>/afcyc_filter_summary_r<round>.csv")
    ap.add_argument("--expected-tasks", type=int, default=None,
                    help="Array tasks per round; reports missing task files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pat = re.compile(rf"afcyc_r{args.round}_task(\d+)\.csv$")
    parts = [p for p in (out_dir / "logs").glob(f"afcyc_r{args.round}_task*.csv") if pat.search(p.name)]
    if not parts:
        sys.exit(f"No logs/afcyc_r{args.round}_task*.csv files under {out_dir}")
    parts.sort(key=lambda p: p.stat().st_mtime)

    rows_by_design = {}
    fieldnames = None
    seen_tasks = set()
    n_rows = n_bad = n_empty = 0

    for part in parts:
        if part.stat().st_size == 0:
            print(f"[WARN] skipping empty file {part.name}", file=sys.stderr)
            n_empty += 1
            continue
        with open(part, newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                print(f"[WARN] skipping {part.name}: no header", file=sys.stderr)
                n_empty += 1
                continue
            if "design" not in reader.fieldnames:
                sys.exit(f"{part.name}: no 'design' column ({reader.fieldnames})")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif set(reader.fieldnames) != set(fieldnames):
                sys.exit(f"{part.name}: column mismatch ({reader.fieldnames} vs {fieldnames})")
            seen_tasks.add(int(pat.search(part.name).group(1)))
            for row in reader:
                # Truncated/half-written rows: extra fields land under key None,
                # missing fields come back as None.
                if None in row or any(v is None for v in row.values()) or not row["design"]:
                    n_bad += 1
                    continue
                n_rows += 1
                rows_by_design[row["design"]] = row

    if fieldnames is None:
        sys.exit("No usable task CSVs found.")

    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"afcyc_filter_summary_r{args.round}.csv"
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for design in sorted(rows_by_design):
            writer.writerow({k: rows_by_design[design].get(k, "") for k in fieldnames})
    os.replace(tmp, out_csv)

    vals = [r.get("passed", "").lower() for r in rows_by_design.values()]
    n_passed, n_err = vals.count("true"), vals.count("error")
    print(f"Merged {len(seen_tasks)} task CSVs ({n_empty} empty skipped) -> "
          f"{len(rows_by_design)} designs: {n_passed} passed, {n_err} errors -> {out_csv}")
    print(f"Rows read: {n_rows}; duplicates overwritten: {n_rows - len(rows_by_design)}; "
          f"malformed rows skipped: {n_bad}")

    if args.expected_tasks:
        missing = sorted(set(range(1, args.expected_tasks + 1)) - seen_tasks)
        if missing:
            shown = ", ".join(map(str, missing[:30])) + (" ..." if len(missing) > 30 else "")
            print(f"[WARN] {len(missing)}/{args.expected_tasks} task files missing "
                  f"(not run, failed, or fully skipped): {shown}", file=sys.stderr)
        else:
            print(f"All {args.expected_tasks} expected task files present.")


if __name__ == "__main__":
    main()
