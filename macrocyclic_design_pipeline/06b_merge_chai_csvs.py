#!/usr/bin/env python3
"""
06b_merge_chai_csvs.py <out-dir> --cboltz-out <05 out dir> [--expected-tasks T]

Writes in <out-dir>:
  chai_summary.csv          all Chai-scored candidates (+ Boltz-vs-AfCycDesign RMSD / ipTM from step 05)
  rescued_hits.csv          Chai agrees with AfCycDesign and/or CyclicBoltz1 (two of three models on one pose)
  chai_errors.csv           candidates with an error (missing Chai output, parsing)
  final_consensus_hits.csv  step-05 consensus hits + rescued hits, with a 'validation' column
  final_consensus_hits.fasta
"""
import argparse
import csv
import glob
import re
from pathlib import Path

VALIDATION = {"chai_agrees_afcyc": "afcyc+chai (boltz disagrees)",
              "chai_agrees_boltz": "boltz+chai (afcyc disagrees)",
              "chai_agrees_both": "afcyc+boltz+chai (all within cutoff of chai)"}


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--cboltz-out", required=True)
    ap.add_argument("--expected-tasks", type=int, default=None)
    a = ap.parse_args()
    out, cb = Path(a.out_dir), Path(a.cboltz_out)

    rows, seen = [], set()
    for f in sorted(glob.glob(str(out / "logs" / "chai_task*.csv"))):
        m = re.search(r"chai_task(\d+)\.csv$", f)
        if not m:
            continue
        seen.add(int(m.group(1)))
        rows += read(f)
    if a.expected_tasks:
        miss = sorted(set(range(1, a.expected_tasks + 1)) - seen)
        if miss:
            print(f"[WARN] {len(miss)} task CSV(s) missing: {miss[:20]}{' ...' if len(miss) > 20 else ''}")
    if not rows:
        raise SystemExit("no Chai result rows found")

    boltz = {r["design"]: r for r in read(cb / "tiebreak_candidates.csv")}
    for r in rows:
        b = boltz.get(r["design"], {})
        r["rmsd_boltz_vs_afcyc"] = b.get("rmsd_binder", "")
        r["iptm_boltz"] = b.get("iptm", "")
    fields = list(rows[0].keys())

    def write(name, subset, cols=None):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols or fields, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(subset)

    ok = [r for r in rows if not r["error"]]
    rescued = sorted((r for r in ok if r["verdict"] in VALIDATION),
                     key=lambda r: -float(r["agg_score"] or 0))
    for r in rescued:
        r["validation"] = VALIDATION[r["verdict"]]
    write("chai_summary.csv", rows)
    write("rescued_hits.csv", rescued, fields + ["validation"])
    write("chai_errors.csv", [r for r in rows if r["error"]])

    final_cols = ["design", "round", "sequence", "validation", "rmsd_boltz_vs_afcyc", "iptm_boltz",
                  "rmsd_chai_vs_afcyc", "rmsd_chai_vs_boltz", "agg_score", "iptm"]
    final = []
    for r in read(cb / "consensus_hits.csv"):
        final.append({"design": r["design"], "round": r["round"], "sequence": r["sequence"],
                      "validation": "afcyc+boltz", "rmsd_boltz_vs_afcyc": r["rmsd_binder"],
                      "iptm_boltz": r["iptm"]})
    n_cons = len(final)
    for r in rescued:
        final.append({**{k: r.get(k, "") for k in final_cols}, "iptm": r["iptm"]})
    write("final_consensus_hits.csv", final, final_cols)
    with open(out / "final_consensus_hits.fasta", "w") as fh:
        for r in final:
            fh.write(f">{r['design']} {r['validation'].split(' ')[0]}\n{r['sequence']}\n")

    from collections import Counter
    c = Counter(r["verdict"] for r in ok)
    print(f"candidates scored {len(rows)} | errors {len(rows) - len(ok)}")
    for k in ("chai_agrees_afcyc", "chai_agrees_boltz", "chai_agrees_both", "no_consensus"):
        print(f"  {k:20s} {c.get(k, 0)}")
    print(f"step-05 consensus {n_cons} + rescued {len(rescued)} = {len(final)} final hits")


if __name__ == "__main__":
    main()
