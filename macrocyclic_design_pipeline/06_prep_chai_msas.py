#!/usr/bin/env python3
"""
06_prep_chai_msas.py  (runs INSIDE the Chai-1 container; needs pandas + pyarrow)

After a `chai-lab fold --use-msa-server` run on prep.fasta, Chai has written one
<sha256(sequence)>.aligned.pqt per protein sequence. This script builds the directory
used with `--msa-directory` for the whole tiebreak run:
  - the target's real MSA (copied),
  - for every candidate binder a query-only MSA, made by copying the query row of the
    prep binder's MSA (same schema) and replacing the sequence.
It verifies the file-naming assumption (sha256 of the sequence) and fails loudly if wrong.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd


def h(seq):
    return hashlib.sha256(seq.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True, help="chai output dir of the prep fold")
    ap.add_argument("--candidates", required=True, help="candidates.tsv (design, sequence, pdb)")
    ap.add_argument("--target-seq", required=True)
    ap.add_argument("--binder-seq", required=True, help="binder sequence of the prep fold")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    found = sorted(Path(a.prep_dir).rglob("*.aligned.pqt"))
    print(f"[*] {len(found)} .aligned.pqt file(s) under {a.prep_dir}")
    by_stem = {f.name[:-len(".aligned.pqt")]: f for f in found}
    tgt_f, bnd_f = by_stem.get(h(a.target_seq)), by_stem.get(h(a.binder_seq))
    if tgt_f is None or bnd_f is None:
        print("[ERROR] expected files named sha256(sequence).aligned.pqt were not found.")
        for f in found:
            try:
                q = pd.read_parquet(f).iloc[0]["sequence"]
            except Exception as e:  # noqa: BLE001
                q = f"<unreadable: {e}>"
            print(f"   {f.name}  query={q[:40]}...  (sha256 of query = {h(q)[:16]}...)")
        sys.exit(1)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tgt_f, out / tgt_f.name)

    tmpl = pd.read_parquet(bnd_f).iloc[[0]].copy()
    print("[*] binder query-row template:\n" + tmpl.to_string(max_colwidth=40))
    if "sequence" not in tmpl.columns:
        sys.exit(f"[ERROR] no 'sequence' column in {bnd_f.name}: {list(tmpl.columns)}")

    seqs = {l.split("\t")[1] for l in open(a.candidates) if l.strip()}
    for s in sorted(seqs):
        row = tmpl.copy()
        row["sequence"] = s
        row.to_parquet(out / f"{h(s)}.aligned.pqt", index=False)
    (out / "READY").write_text("ok\n")
    print(f"[*] wrote target MSA + {len(seqs)} query-only binder MSAs to {out}")


if __name__ == "__main__":
    main()
