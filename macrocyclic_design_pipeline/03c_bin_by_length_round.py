#!/usr/bin/env python3
"""03c_bin_by_length_round.py OUT_DIR [N_ROUNDS=4] [--csv prefix]

OUT_DIR = e.g. 03_mpnn_outputs/Mcl1
Success  = round<r>/seqs/*.fa exists (failed rounds are deleted); length = chain A of the accepted sequence.
Failure  = OUT_DIR/fail_reports/*.tsv (stem, round, pep_len, ...), if present.
"""
import sys, os, glob, csv, collections

out = sys.argv[1].rstrip("/")
nr = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
csv_prefix = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None

succ = collections.defaultdict(int)         # (length, round) -> accepted
fail = collections.defaultdict(int)         # (length, round) -> failed
seen = collections.defaultdict(set)         # length -> stems that entered the pipeline
done = collections.defaultdict(set)         # stem -> rounds accepted
stem_len = {}

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
        stem_len[stem] = n
        seen[n].add(stem)

for tsv in glob.glob(f"{out}/fail_reports/*.tsv"):
    f = open(tsv).readline().rstrip("\n").split("\t")
    stem, r, n = f[0], int(f[1]), int(f[2])
    fail[(n, r)] += 1
    seen[n].add(stem)

lengths = sorted(seen)
full = set(range(1, nr + 1))
rows = []
for n in lengths:
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
hdr = ["len", "backbones"] + [f"R{r} ok/fail (%)" for r in range(1, nr + 1)] + ["all rounds"]
print("\t".join(hdr))
for x in rows:
    cells = [str(x["length"]), str(x["backbones"])]
    for r in range(1, nr + 1):
        cells.append(f'{x[f"r{r}_ok"]}/{x[f"r{r}_fail"]} ({x[f"r{r}_rate"]}%)' if x[f"r{r}_rate"] != "" else "-")
    cells.append(f'{x["completed_all"]} ({x["completed_pct"]}%)')
    print("\t".join(cells))

if csv_prefix:
    with open(f"{csv_prefix}_by_length_round.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)