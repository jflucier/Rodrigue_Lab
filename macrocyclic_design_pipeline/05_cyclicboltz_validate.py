#!/usr/bin/env python3
"""
05_cyclicboltz_validate.py  (runs INSIDE the CyclicBoltz1 container)

For every design in --jobs-tsv (design<TAB>afcyc_input_pdb):
  1. take binder + target sequences from the PDB (binder chain is cyclic),
  2. write a Boltz YAML (binder cyclic, target single chain),
  3. run `boltz predict` once per sub-batch (model loaded once per batch),
  4. compare the Boltz pose with the AfCycDesign pose:
       rmsd_binder = binder CA RMSD after superposing on the target CA
                     (same definition as the AfCycDesign 'rmsd' column),
  5. append a row to logs/cboltz_task<task-id>.csv  (resumable).

Reference pose: first file matching <afcyc-pred-dir>/<design>*.pdb if
--afcyc-pred-dir is given, otherwise the relaxed input complex (which AfCycDesign
reproduced to < --rmsd-cutoff of 04, so this is a close proxy; the column
ref_source records which was used).
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}

COLS = ["design", "round", "sequence", "rmsd_binder", "rmsd_target", "agree",
        "iptm", "ptm", "confidence_score", "complex_plddt", "complex_iplddt",
        "ref_source", "error"]


def read_ca(pdb):
    """{chain: [(aa1, xyz), ...]} from CA atoms (first altloc only)."""
    chains = {}
    with open(pdb) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            chains.setdefault(line[21], []).append((AA3.get(line[17:20].strip(), "X"), xyz))
    return chains


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, pc, qc


def rmsd(A, B):
    return float(np.sqrt(((A - B) ** 2).sum(1).mean()))


def compare(pred_pdb, ref_pdb, bch, tch):
    p, r = read_ca(pred_pdb), read_ca(ref_pdb)
    for ch in (bch, tch):
        if ch not in p or ch not in r:
            raise ValueError(f"chain {ch} missing (pred {sorted(p)}, ref {sorted(r)})")
        if len(p[ch]) != len(r[ch]):
            raise ValueError(f"chain {ch}: {len(p[ch])} CA in prediction vs {len(r[ch])} in reference")
    xyz = lambda d, c: np.array([a[1] for a in d[c]])
    Pt, Qt = xyz(p, tch), xyz(r, tch)
    R, pc, qc = kabsch(Pt, Qt)               # move prediction onto reference via target
    Pb = (xyz(p, bch) - pc) @ R.T + qc
    return rmsd(Pb, xyz(r, bch)), rmsd((Pt - pc) @ R.T + qc, Qt)


def write_yaml(path, binder_seq, target_seq, bch, tch, target_msa):
    # CyclicBoltz1's relative-position encoder applies the cyclic offset to the LAST
    # chain of the complex (l2 = residue_index[0,-1] + 1), so the cyclic binder must be
    # listed last; the chain IDs are unaffected.
    path.write_text(
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: {tch}\n      sequence: {target_seq}\n"
        f"      msa: {target_msa}\n"
        f"  - protein:\n      id: {bch}\n      sequence: {binder_seq}\n"
        f"      msa: empty\n      cyclic: true\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--binder-chain", default="A")
    ap.add_argument("--target-chain", default="B")
    ap.add_argument("--rmsd-cutoff", type=float, default=2.0)
    ap.add_argument("--afcyc-pred-dir", default=None)
    ap.add_argument("--target-msa", default="empty", help="a3m/csv for the target, or 'empty'")
    ap.add_argument("--boltz-cmd", default="boltz")
    ap.add_argument("--boltz-cache", default=None)
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--diffusion-samples", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=25, help="designs per `boltz predict` call")
    ap.add_argument("--scratch-dir", default=None, help="scratch root (must be visible in the container)")
    ap.add_argument("--extra-args", default="", help="extra flags appended to `boltz predict`")
    a = ap.parse_args()

    logs = Path(a.out_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    csv_path = logs / f"cboltz_task{a.task_id}.csv"

    jobs = [l.rstrip("\n").split("\t") for l in open(a.jobs_tsv) if l.strip()]
    done = set()
    if csv_path.exists():
        with open(csv_path) as fh:
            done = {r["design"] for r in csv.DictReader(fh)}
    jobs = [j for j in jobs if j[0] not in done]
    print(f"[*] task {a.task_id}: {len(jobs)} designs to run ({len(done)} already done)", flush=True)

    new = not csv_path.exists()
    out = open(csv_path, "a", newline="")
    w = csv.DictWriter(out, fieldnames=COLS)
    if new:
        w.writeheader()

    # SLURM's TMPDIR (/localscratch/...) is usually not bound into the container,
    # so only use it if it is visible here; otherwise scratch under the output dir.
    scratch_root = a.scratch_dir or os.environ.get("TMPDIR") or ""
    if not scratch_root or not os.path.isdir(scratch_root) or not os.access(scratch_root, os.W_OK):
        scratch_root = str(Path(a.out_dir) / "tmp")
    os.makedirs(scratch_root, exist_ok=True)
    print(f"[*] scratch: {scratch_root}", flush=True)
    for b0 in range(0, len(jobs), a.batch_size):
        batch = jobs[b0:b0 + a.batch_size]
        tmp = Path(tempfile.mkdtemp(prefix=f"cboltz_{a.task_id}_", dir=scratch_root))
        inp, res = tmp / "inputs", tmp / "res"
        inp.mkdir()
        rows, meta = {}, {}
        for design, pdb in batch:
            row = {c: "" for c in COLS}
            row["design"] = design
            m = design.rsplit("_r", 1)
            row["round"] = m[1] if len(m) == 2 and m[1].isdigit() else ""
            rows[design] = row
            try:
                ca = read_ca(pdb)
                bseq = "".join(x[0] for x in ca[a.binder_chain])
                tseq = "".join(x[0] for x in ca[a.target_chain])
                if "X" in bseq or "X" in tseq:
                    raise ValueError("non-standard residue in input")
                row["sequence"] = bseq
                write_yaml(inp / f"{design}.yaml", bseq, tseq, a.binder_chain,
                           a.target_chain, a.target_msa)
                meta[design] = pdb
            except Exception as e:  # noqa: BLE001
                row["error"] = f"input: {e}"

        if meta:
            cmd = [a.boltz_cmd, "predict", str(inp), "--out_dir", str(res),
                   "--output_format", "pdb", "--override",
                   "--recycling_steps", str(a.recycling_steps),
                   "--sampling_steps", str(a.sampling_steps),
                   "--diffusion_samples", str(a.diffusion_samples)]
            if a.boltz_cache:
                cmd += ["--cache", a.boltz_cache]
            if a.extra_args:
                cmd += a.extra_args.split()
            print("[*] " + " ".join(cmd), flush=True)
            # torch>=2.6 defaults to weights_only=True, which cannot unpickle the official
            # Boltz-1 Lightning checkpoint (omegaconf DictConfig). The checkpoint is the
            # official Boltz download, so force the legacy behaviour for this subprocess.
            env = dict(os.environ)
            env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            rc = subprocess.run(cmd, env=env).returncode
            n_pred = len(glob.glob(str(res / "**" / "predictions" / "*" / "*_model_0.pdb"), recursive=True))
            if rc != 0:
                print(f"[WARN] boltz exited {rc}; {n_pred} predictions written", flush=True)
            if rc != 0 and n_pred == 0:
                # systematic failure: do not write error rows (they would be skipped on resume)
                out.close()
                shutil.rmtree(tmp, ignore_errors=True)
                sys.exit(f"[ERROR] boltz failed with no output for batch starting at {b0}; "
                         f"nothing written to {csv_path} for it")

        for design, pdb in meta.items():
            row = rows[design]
            try:
                hits = glob.glob(str(res / "**" / "predictions" / design / f"{design}_model_0.pdb"),
                                 recursive=True)
                if not hits:
                    raise FileNotFoundError("no Boltz prediction written")
                ref, src = pdb, "relaxed_input"
                if a.afcyc_pred_dir:
                    c = sorted(glob.glob(str(Path(a.afcyc_pred_dir) / f"{design}*.pdb")))
                    if c:
                        ref, src = c[0], "afcyc_prediction"
                row["ref_source"] = src
                rb, rt = compare(hits[0], ref, a.binder_chain, a.target_chain)
                row["rmsd_binder"], row["rmsd_target"] = f"{rb:.4f}", f"{rt:.4f}"
                row["agree"] = rb < a.rmsd_cutoff
                cj = glob.glob(str(res / "**" / "predictions" / design / f"confidence_{design}_model_0.json"),
                               recursive=True)
                if cj:
                    conf = json.load(open(cj[0]))
                    for k in ("iptm", "ptm", "confidence_score", "complex_plddt", "complex_iplddt"):
                        if k in conf:
                            row[k] = f"{conf[k]:.4f}"
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{e}"
        for row in rows.values():
            w.writerow(row)
        out.flush()
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[*] wrote {b0 + len(batch)}/{len(jobs)}", flush=True)
    out.close()


if __name__ == "__main__":
    sys.exit(main())
