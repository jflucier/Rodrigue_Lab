#!/usr/bin/env python3
"""
06_chai_score.py  (CPU; runs in the Chai container, needs only numpy + stdlib)

For each design in --jobs-tsv (design<TAB>binder_seq<TAB>relaxed_pdb) reads the Chai-1 samples in
<chai-dir>/<design>/ (pred.model_idx_N.cif + scores.model_idx_N.npz), and for every sample computes the
binder CA RMSD (superposed on the target CA, same definition as step 05) against
  - the AfCycDesign pose (<afcyc-pred-dir>/**/<design>*.pdb, else the relaxed input complex),
  - the CyclicBoltz1 pose (<boltz-pred-dir>/<design>_model_0.pdb).
The top-ranked sample (highest Chai aggregate_score) decides the verdict:
  chai_agrees_both | chai_agrees_afcyc | chai_agrees_boltz | no_consensus
Chains in the Chai CIF are identified by sequence, not by ID.
"""
import argparse
import csv
import glob
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cbv", HERE / "05_cyclicboltz_validate.py")
cbv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cbv)
AA1TO3 = {v: k for k, v in cbv.AA3.items()}

COLS = ["design", "round", "sequence", "n_samples", "top_sample", "agg_score", "iptm", "ptm",
        "rmsd_chai_vs_afcyc", "rmsd_chai_vs_boltz", "rmsd_target_vs_afcyc",
        "n_samples_agree_afcyc", "n_samples_agree_boltz", "agree_afcyc", "agree_boltz",
        "verdict", "ref_source", "error"]


def read_cif_chains(path):
    """{label_asym_id: [(aa1, (x, y, z)), ...]} from CA atoms of the first model."""
    chains, hdr, state, idx = {}, [], None, {}
    with open(path) as fh:
        for line in fh:
            t = line.strip()
            if t == "loop_":
                hdr, state = [], "hdr"
                continue
            if state == "hdr":
                if t.startswith("_"):
                    hdr.append(t)
                    continue
                state = "data" if hdr and hdr[0].startswith("_atom_site.") else None
                if state == "data":
                    idx = {h: i for i, h in enumerate(hdr)}
            if state == "data":
                if not t or t.startswith(("#", "loop_", "data_", "_")):
                    state = None
                    continue
                f = t.split()
                if len(f) < len(hdr) or f[idx["_atom_site.label_atom_id"]] != "CA":
                    continue
                if "_atom_site.group_PDB" in idx and f[idx["_atom_site.group_PDB"]] != "ATOM":
                    continue
                if f[idx.get("_atom_site.pdbx_PDB_model_num", idx["_atom_site.label_atom_id"])] not in ("1", "CA"):
                    continue
                aa = cbv.AA3.get(f[idx["_atom_site.label_comp_id"]], "X")
                xyz = tuple(float(f[idx[k]]) for k in ("_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"))
                chains.setdefault(f[idx["_atom_site.label_asym_id"]], []).append((aa, xyz))
    return chains


def cif_to_ca_pdb(cif, binder_seq, pdb_out):
    """Write a CA-only PDB with chain A = binder, chain B = target (identified by sequence)."""
    ch = read_cif_chains(cif)
    if len(ch) < 2:
        raise ValueError(f"Chai CIF has chains {sorted(ch)}; expected binder + target")
    bid = max(ch, key=lambda c: cbv._sim(binder_seq, cbv._seq(ch[c])))
    tid = max((c for c in ch if c != bid), key=lambda c: len(ch[c]))
    lines, n = [], 0
    for cid, key in ((bid, "A"), (tid, "B")):
        for i, (aa, xyz) in enumerate(ch[cid], 1):
            n += 1
            lines.append("ATOM  %5d  CA  %3s %1s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C"
                         % (n, AA1TO3.get(aa, "ALA"), key, i, *xyz))
    Path(pdb_out).write_text("\n".join(lines) + "\nEND\n")


def npz_val(d, key):
    try:
        return float(np.ravel(d[key])[0])
    except Exception:  # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-tsv", required=True)
    ap.add_argument("--chai-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--boltz-pred-dir", required=True)
    ap.add_argument("--afcyc-pred-dir", default=None)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--rmsd-cutoff", type=float, default=2.0)
    a = ap.parse_args()

    jobs = [l.rstrip("\n").split("\t") for l in open(a.jobs_tsv) if l.strip()]
    tmpdir = tempfile.mkdtemp(prefix="chai_score_", dir=str(Path(a.out_csv).parent))
    rows = []
    for design, bseq, relaxed in jobs:
        row = {c: "" for c in COLS}
        row["design"], row["sequence"] = design, bseq
        m = design.rsplit("_r", 1)
        row["round"] = m[1] if len(m) == 2 and m[1].isdigit() else ""
        try:
            cifs = sorted(glob.glob(os.path.join(a.chai_dir, design, "pred.model_idx_*.cif")))
            if not cifs:
                raise FileNotFoundError("no Chai output")
            ref_af, row["ref_source"] = relaxed, "relaxed_input"
            if a.afcyc_pred_dir:
                c = sorted(glob.glob(os.path.join(a.afcyc_pred_dir, "**", f"{design}*.pdb"), recursive=True))
                if c:
                    ref_af, row["ref_source"] = c[0], "afcyc_prediction"
            bp = Path(a.boltz_pred_dir) / f"{design}_model_0.pdb"
            ref_bo = str(bp) if bp.is_file() else None

            samples = []
            for cif in cifs:
                k = Path(cif).stem.split("_")[-1]
                sc = Path(cif).with_name(f"scores.model_idx_{k}.npz")
                d = np.load(sc) if sc.is_file() else {}
                tmp_pdb = os.path.join(tmpdir, f"{design}_{k}.pdb")
                cif_to_ca_pdb(cif, bseq, tmp_pdb)
                r_af, t_af = cbv.compare(tmp_pdb, ref_af, "A", "B")
                r_bo = cbv.compare(tmp_pdb, ref_bo, "A", "B")[0] if ref_bo else None
                os.remove(tmp_pdb)
                samples.append({"agg": npz_val(d, "aggregate_score") if len(d) else float("nan"),
                                "iptm": npz_val(d, "iptm") if len(d) else float("nan"),
                                "ptm": npz_val(d, "ptm") if len(d) else float("nan"),
                                "r_af": r_af, "t_af": t_af, "r_bo": r_bo, "k": k})
            aggs = [s["agg"] for s in samples]
            top = samples[int(np.nanargmax(aggs))] if not all(np.isnan(aggs)) else samples[0]
            row.update(n_samples=len(samples), top_sample=top["k"],
                       agg_score=f"{top['agg']:.4f}", iptm=f"{top['iptm']:.4f}", ptm=f"{top['ptm']:.4f}",
                       rmsd_chai_vs_afcyc=f"{top['r_af']:.4f}", rmsd_target_vs_afcyc=f"{top['t_af']:.4f}",
                       n_samples_agree_afcyc=sum(s["r_af"] < a.rmsd_cutoff for s in samples))
            ag_af = top["r_af"] < a.rmsd_cutoff
            ag_bo = None
            if top["r_bo"] is not None:
                row["rmsd_chai_vs_boltz"] = f"{top['r_bo']:.4f}"
                row["n_samples_agree_boltz"] = sum(s["r_bo"] < a.rmsd_cutoff for s in samples)
                ag_bo = top["r_bo"] < a.rmsd_cutoff
            row["agree_afcyc"], row["agree_boltz"] = ag_af, ("" if ag_bo is None else ag_bo)
            row["verdict"] = ("chai_agrees_both" if ag_af and ag_bo else "chai_agrees_afcyc" if ag_af
                              else "chai_agrees_boltz" if ag_bo else "no_consensus")
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)
        rows.append(row)

    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
    n_err = sum(1 for r in rows if r["error"])
    print(f"[*] scored {len(rows)} designs ({n_err} errors) -> {a.out_csv}")


if __name__ == "__main__":
    sys.exit(main())
