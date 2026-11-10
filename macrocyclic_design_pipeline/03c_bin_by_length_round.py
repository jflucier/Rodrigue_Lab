#!/usr/bin/env python3
"""
Bin accepted MPNN sequences by peptide length vs round number.

Usage:
  python 03c_bin_by_length_round.py \
      --out-dir /path/to/03_mpnn_outputs/Mcl1 --n-rounds 4 --csv_prefix Mcl1

Success = <out-dir>/<stem>/round<r>/seqs/*.fa exists (failed rounds are deleted);
          length = chain A of the accepted sequence.
Failure = <out-dir>/fail_reports/*.tsv (columns: stem, round, pep_len, ...), if present.
"""
import argparse
import collections
import csv
import glob
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="MPNN output directory for one target (e.g. .../03_mpnn_outputs/Mcl1)")
    p.add_argument("--n-rounds", type=int, default=4, help="Number of design rounds (default: 4)")
    p.add_argument("--csv_prefix", "--csv-prefix", dest="csv_prefix", default=None,
                   help="If set, write <prefix>_by_length_round.csv in the current directory")
    return p.parse_args()


def main():
    a = parse_args()
    out = a.out_dir.rstrip("/")
    nr = a.n_rounds
    if not os.path.isdir(out):
        sys.exit(f"[ERROR] out-dir not found: {out}")

    succ = collections.defaultdict(int)   # (length, round) -> accepted
    fail = collections.defaultdict(int)   # (length, round) -> failed
    seen = collections.defaultdict(set)   # length -> stems that entered the pipeline
    done = collections.defaultdict(set)   # stem -> rounds accepted

    for d in glob.glob(f"{out}/diffused_binder_cyclic_*/"):
        stem = os.path.basename(d.rstrip("/"))
        for r in range(1, nr + 1):
            fas = glob.glob(f"{d}round{r}/seqs/*.fa")
            if not fas:
                continue
            lines = open(fas[0]).read().split("\n")
            if len(lines) < 4 or not lines[3]:
                continue
            n = len(lines[3].split("/")[0])
            succ[(n, r)] += 1
            done[stem].add(r)
            seen[n].add(stem)

    n_fail_reports = 0
    for tsv in glob.glob(f"{out}/fail_reports/*.tsv"):
        f = open(tsv).readline().rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        stem, r, n = f[0], int(f[1]), int(f[2])
        fail[(n, r)] += 1
        seen[n].add(stem)
        n_fail_reports += 1

    if not seen:
        sys.exit(f"[ERROR] no accepted sequences or failure reports found under {out}")
    if n_fail_reports == 0:
        print("[NOTE] no fail_reports/ found: percentages only cover lineages that succeeded at least once "
              "and overstate pass rates.", file=sys.stderr)

    full = set(range(1, nr + 1))
    rows = []
    for n in sorted(seen):
        row = {"length": n, "backbones": len(seen[n])}
        for r in range(1, nr + 1):
            s, f_ = succ[(n, r)], fail[(n, r)]
            row[f"r{r}_ok"], row[f"r{r}_fail"] = s, f_
            row[f"r{r}_rate"] = round(100 * s / (s + f_), 1) if (s + f_) else ""
        comp = sum(1 for st in seen[n] if done.get(st, set()) >= full)
        row["completed_all"] = comp
        row["completed_pct"] = round(100 * comp / len(seen[n]), 1)
        rows.append(row)

    # console table
    print("\t".join(["len", "backbones"] + [f"R{r} ok/fail (%)" for r in range(1, nr + 1)] + ["all rounds"]))
    for x in rows:
        cells = [str(x["length"]), str(x["backbones"])]
        for r in range(1, nr + 1):
            rate = x[f"r{r}_rate"]
            cells.append(f'{x[f"r{r}_ok"]}/{x[f"r{r}_fail"]} ({rate}%)' if rate != "" else "-")
        cells.append(f'{x["completed_all"]} ({x["completed_pct"]}%)')
        print("\t".join(cells))

    if a.csv_prefix:
        path = f"{a.csv_prefix}_by_length_round.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] wrote {path}", file=sys.stderr)

if __name__ == "__main__":
    main()