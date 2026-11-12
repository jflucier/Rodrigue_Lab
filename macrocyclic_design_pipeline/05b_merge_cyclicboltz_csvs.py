#!/usr/bin/env python3
"""
05b_merge_cyclicboltz_csvs.py <out-dir> [--expected-tasks T]

Writes in <out-dir>:
  cboltz_summary.csv         all rows
  consensus_hits.csv         agree==True (AfCycDesign + CyclicBoltz1 agree on pose)
  tiebreak_candidates.csv    agree==False, no error (candidates for Chai-1/AF3 tiebreak)
  cboltz_errors.csv          rows with an error
  consensus_hits.fasta       binder sequences of consensus hits
"""
import argparse
import csv
import glob
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--expected-tasks", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out_dir)

    rows, fields, seen = [], None, set()
    for f in sorted(glob.glob(str(out / "logs" / "cboltz_task*.csv"))):
        m = re.search(r"cboltz_task(\d+)\.csv$", f)
        if not m or int(m.group(1)) >= 100000:
            continue
        seen.add(int(m.group(1)))
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh)
            fields = fields or rd.fieldnames
            rows += list(rd)
    if a.expected_tasks:
        miss = sorted(set(range(1, a.expected_tasks + 1)) - seen)
        if miss:
            print(f"[WARN] {len(miss)} task CSV(s) missing: {miss[:20]}{' ...' if len(miss) > 20 else ''}")
    if not rows:
        raise SystemExit("no rows found")

    def write(name, subset):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    ok = [r for r in rows if not r["error"]]
    cons = sorted((r for r in ok if r["agree"] == "True"), key=lambda r: float(r["rmsd_binder"]))
    dis = [r for r in ok if r["agree"] != "True"]
    write("cboltz_summary.csv", rows)
    write("consensus_hits.csv", cons)
    write("tiebreak_candidates.csv", dis)
    write("cboltz_errors.csv", [r for r in rows if r["error"]])
    with open(out / "consensus_hits.fasta", "w") as fh:
        for r in cons:
            fh.write(f">{r['design']} rmsd={r['rmsd_binder']} iptm={r['iptm']}\n{r['sequence']}\n")

    print(f"rows {len(rows)} | consensus {len(cons)} | disagree {len(dis)} | errors {len(rows) - len(ok)}")
    if ok:
        print(f"agreement rate (of successful predictions): {100 * len(cons) / len(ok):.1f}%")


if __name__ == "__main__":
    main()
