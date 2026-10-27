#!/usr/bin/env python3
"""
04d_check_rmsd.py

Independent check of the `rmsd` column written by 04_afcycdesign_filter.py.
CPU only, no ColabDesign needed: it re-reads the saved predictions and the
relaxed input complexes and computes, per predicted model:

  bind_tgtalign   Ca RMSD of the BINDER after superposing the prediction on the
                  input using the TARGET Ca atoms only (the usual "peptide RMSD
                  aligned on the target"; this is what a binder RMSD should be)
  bind_ifacealign Ca RMSD of the BINDER after superposing on only the target
                  residues near the binder (CA within --interface-cutoff A of
                  any binder CA in the input). Robust to the target flexing or
                  being refolded away from the interface.
  iface_recall    fraction of the DESIGNED interface (target residues near the binder
                  in the input) still contacted by the predicted binder (prediction
                  fitted on the target). ~1 = right site; ~0 = binder predicted elsewhere.
  iface_jaccard   same overlap as intersection/union (also penalises extra contacts)
  (full run) also writes --contact-out: for every target residue, how often it is in the
                  DESIGNED interface vs how often the PREDICTED binder contacts it
                  (all designs, and designs with iPAE < --ipae-cutoff-hist), to show
                  whether AlphaFold funnels binders to one particular surface.
  complex_aligned Ca RMSD of target+binder after superposing on ALL Ca atoms
                  (what `align` + `rmsd` on the whole complex gives in ChimeraX)
  complex_raw     Ca RMSD of target+binder with NO superposition
                  (what `rmsd` without `align` gives in ChimeraX)

and compares each with the ColabDesign value, to show which quantity
ColabDesign's `rmsd` actually is.

Usage (from the project directory):
  python3 04d_check_rmsd.py \
      --csv '04_afcycdesign_outputs/dnaJ/logs/afcyc_r*_task*.csv' \
      --predictions-dir 04_afcycdesign_outputs/dnaJ/predictions \
      --relax-out 03_mpnn_outputs/dnaJ \
      --out rmsd_check.csv

  # one design in detail, to compare with ChimeraX:
  python3 04d_check_rmsd.py ... --design diffused_binder_cyclic_2_r4

Input complex = <relax-out>/<backbone>/round<N>/<design>.pdb (design = <backbone>_r<N>),
prediction    = <predictions-dir>/<design>_prediction.pdb (one MODEL per model run).
Chains: in the input the binder is --binder-chain (A) and the target --target-chain (B);
in the prediction chains are identified by length (binder = the shorter chain), so the
ColabDesign output order (target first) does not matter.
"""
import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- IO
def read_ca_models(path):
    """Return a list of models; each model is {chain_id: (n,3) array of CA coords in file order}."""
    models, cur = [], None
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec == "MODEL ":
                cur = {}
            elif rec == "ENDMDL":
                if cur:
                    models.append(cur)
                cur = None
            elif rec == "ATOM  " and line[12:16] == " CA " and line[16] in (" ", "A", "1"):
                if cur is None:
                    cur = {}
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                cur.setdefault(line[21], []).append(xyz)
    if cur:
        models.append(cur)
    return [{c: np.asarray(v, dtype=float) for c, v in m.items()} for m in models]


def split_binder_target(chains, binder_hint=None, target_hint=None):
    """Return (binder, target) CA arrays. Uses the hints if both exist, else binder = shorter chain."""
    if binder_hint in chains and target_hint in chains:
        return chains[binder_hint], chains[target_hint]
    if len(chains) != 2:
        raise ValueError(f"expected 2 chains, found {sorted(chains)}")
    a, b = sorted(chains.values(), key=len)
    return a, b


# ------------------------------------------------------------------------ math
def fit(P, Q):
    """Rigid transform (Kabsch) mapping P onto Q; returns a function applying it."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return lambda X: (X - Pc) @ R.T + Qc


def rmsd(A, B):
    return float(np.sqrt(((A - B) ** 2).sum(1).mean()))


def interface_idx(in_b, in_t, cutoff):
    """Indices of target CA atoms within `cutoff` A of any binder CA (at least 6 nearest)."""
    d = np.sqrt(((in_t[:, None, :] - in_b[None, :, :]) ** 2).sum(-1)).min(1)
    idx = np.where(d < cutoff)[0]
    if len(idx) < 6:
        idx = np.argsort(d)[: min(6, len(d))]
    return idx


def model_metrics(pred_b, pred_t, in_b, in_t, iface, contact_cutoff=8.0, in_contact=None):
    both_p, both_i = np.vstack([pred_t, pred_b]), np.vstack([in_t, in_b])
    f_t = fit(pred_t, in_t)
    f_i = fit(pred_t[iface], in_t[iface])
    f_all = fit(both_p, both_i)
    d_pred = np.sqrt(((in_t[:, None, :] - f_t(pred_b)[None, :, :]) ** 2).sum(-1)).min(1)
    pi, di = set(np.where(d_pred < contact_cutoff)[0].tolist()), set(in_contact.tolist())
    return {
        "_pred_contact": (d_pred < contact_cutoff),
        "iface_recall": len(pi & di) / max(len(di), 1),
        "iface_jaccard": len(pi & di) / max(len(pi | di), 1),
        "bind_tgtalign": rmsd(f_t(pred_b), in_b),
        "bind_ifacealign": rmsd(f_i(pred_b), in_b),
        "iface_fit_rmsd": rmsd(f_i(pred_t[iface]), in_t[iface]),
        "_tdev": np.linalg.norm(f_t(pred_t) - in_t, axis=1),
        "target_after_tgtfit": rmsd(f_t(pred_t), in_t),
        "complex_aligned": rmsd(f_all(both_p), both_i),
        "complex_raw": rmsd(both_p, both_i),
    }


def target_resids(pdb_path, binder_chain, target_chain, n_t):
    """Residue numbers (as in the input PDB) of the target chain CA atoms, in file order."""
    out = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line[:6] == "ATOM  " and line[12:16] == " CA " and line[16] in (" ", "A", "1"):
                out.setdefault(line[21], []).append(line[22:27].strip())
    if target_chain in out and len(out[target_chain]) == n_t:
        return out[target_chain]
    for v in out.values():
        if len(v) == n_t:
            return v
    return [str(i + 1) for i in range(n_t)]


def design_paths(design, relax_out, pred_dir):
    m = re.match(r"(.+)_r(\d+)$", design)
    if not m:
        raise ValueError(f"cannot split design name {design!r} into <backbone>_r<round>")
    backbone, rnd = m.group(1), m.group(2)
    return (Path(relax_out) / backbone / f"round{rnd}" / f"{design}.pdb",
            Path(pred_dir) / f"{design}_prediction.pdb")


def analyse(design, relax_out, pred_dir, binder_chain, target_chain, iface_cutoff=10.0, contact_cutoff=8.0):
    in_path, pr_path = design_paths(design, relax_out, pred_dir)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)
    if not pr_path.is_file():
        raise FileNotFoundError(pr_path)
    in_models = read_ca_models(in_path)
    pr_models = read_ca_models(pr_path)
    if not in_models or not pr_models:
        raise ValueError("no CA atoms parsed")
    in_b, in_t = split_binder_target(in_models[0], binder_chain, target_chain)
    iface = interface_idx(in_b, in_t, iface_cutoff)
    d_in = np.sqrt(((in_t[:, None, :] - in_b[None, :, :]) ** 2).sum(-1)).min(1)
    in_contact = np.where(d_in < contact_cutoff)[0]
    per_model = []
    for m in pr_models:
        pb, pt = split_binder_target(m)
        if len(pb) != len(in_b) or len(pt) != len(in_t):
            raise ValueError(f"CA counts differ: prediction binder/target {len(pb)}/{len(pt)}, "
                             f"input {len(in_b)}/{len(in_t)}")
        per_model.append(model_metrics(pb, pt, in_b, in_t, iface, contact_cutoff, in_contact))
    return per_model, iface, in_contact, in_path


# ------------------------------------------------------------------------ main
def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="afcyc_r*_task*.csv",
                    help="glob of task CSVs written by 04_afcycdesign_filter.py (quote it)")
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--relax-out", required=True, help="03 output dir (holds <backbone>/round<N>/*.pdb)")
    ap.add_argument("--binder-chain", default="A")
    ap.add_argument("--target-chain", default="B")
    ap.add_argument("--interface-cutoff", type=float, default=10.0,
                    help="target CA within this distance (A) of a binder CA define the interface "
                         "used for the interface-aligned binder RMSD")
    ap.add_argument("--contact-cutoff", type=float, default=8.0,
                    help="CA-CA distance (A) defining a target residue as contacted by the binder, "
                         "used for the site-recovery metric (iface_recall / iface_jaccard)")
    ap.add_argument("--contact-out", default="target_contact_frequency.csv")
    ap.add_argument("--ipae-cutoff-hist", type=float, default=0.40,
                    help="iPAE cutoff for the 'confident' column of --contact-out")
    ap.add_argument("--ipae-cutoff", type=float, default=0.30)
    ap.add_argument("--rmsd-cutoff", type=float, default=1.5)
    ap.add_argument("--max", type=int, default=None, help="only analyse the first N designs")
    ap.add_argument("--design", default=None, help="analyse and print this design only")
    ap.add_argument("--out", default="rmsd_check.csv")
    args = ap.parse_args()

    rows = {}
    for f in sorted(glob.glob(args.csv)):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("design") and r.get("passed") in ("True", "False"):
                    rows[r["design"]] = r
    if args.design:
        rows = {args.design: rows.get(args.design, {"design": args.design})}
    if not rows:
        sys.exit(f"no usable rows found with --csv {args.csv!r}")
    names = list(rows)[: args.max] if args.max else list(rows)
    print(f"[*] {len(names)} designs to analyse")

    out_rows, n_fail = [], 0
    des_cnt = pred_cnt = pred_cnt_conf = None
    n_hist = n_hist_conf = 0
    resids = None
    for k, name in enumerate(names, 1):
        try:
            pm, iface, in_contact, in_path = analyse(name, args.relax_out, args.predictions_dir, args.binder_chain,
                                args.target_chain, args.interface_cutoff, args.contact_cutoff)
        except Exception as e:  # keep going; report at the end
            n_fail += 1
            if n_fail <= 5:
                print(f"  [WARN] {name}: {e}")
            continue
        r = rows[name]
        if not args.design:
            n_t = len(pm[0]["_pred_contact"])
            if des_cnt is None:
                des_cnt, pred_cnt, pred_cnt_conf = np.zeros(n_t), np.zeros(n_t), np.zeros(n_t)
                resids = target_resids(in_path, args.binder_chain, args.target_chain, n_t)
            if n_t == len(des_cnt):
                dv = np.zeros(n_t)
                dv[in_contact] = 1.0
                pv = np.mean([m["_pred_contact"] for m in pm], axis=0)
                des_cnt += dv
                pred_cnt += pv
                n_hist += 1
                if fnum(r.get("ipae")) < args.ipae_cutoff_hist:
                    pred_cnt_conf += pv
                    n_hist_conf += 1
        o = {"design": name, "n_models": len(pm),
             "rmsd_colabdesign": fnum(r.get("rmsd")), "ipae": fnum(r.get("ipae")),
             "plddt": fnum(r.get("plddt"))}
        o["n_iface"] = len(iface)
        for key in ("bind_tgtalign", "bind_ifacealign", "iface_fit_rmsd", "iface_recall", "iface_jaccard",
                    "complex_aligned",
                    "complex_raw", "target_after_tgtfit"):
            vals = [m[key] for m in pm]
            o[f"{key}_mean"] = float(np.mean(vals))
            o[f"{key}_per_model"] = ";".join(f"{v:.3f}" for v in vals)
        o["bind_tgtalign_min"] = float(min(m["bind_tgtalign"] for m in pm))
        o["bind_ifacealign_min"] = float(min(m["bind_ifacealign"] for m in pm))
        out_rows.append(o)
        if args.design:
            print(f"\n{name}: ColabDesign rmsd={o['rmsd_colabdesign']}  ipae={o['ipae']}")
            for i, m in enumerate(pm):
                print(f"  model {i + 1}: binder after target fit {m['bind_tgtalign']:.3f} | "
                      f"binder after interface fit {m['bind_ifacealign']:.3f} "
                      f"(interface fit {m['iface_fit_rmsd']:.2f}) | "
                      f"designed-interface recall {m['iface_recall']:.2f} | "
                      f"complex aligned {m['complex_aligned']:.3f} | complex raw {m['complex_raw']:.3f} | "
                      f"target after target fit {m['target_after_tgtfit']:.3f}")
            dev = pm[0]["_tdev"]
            far = [(i + 1, round(float(x), 1)) for i, x in enumerate(dev) if x > 3.0]
            print(f"  interface target residues (1-based, {len(iface)}): {[int(i) + 1 for i in iface]}")
            print(f"  target CA deviation after target fit, model 1: median {np.median(dev):.2f} A, "
                  f"max {dev.max():.2f} A")
            print(f"  residues deviating > 3 A (index, A): {far if far else 'none'}")
            print("  per-residue deviation (A): " + " ".join(f"{x:.1f}" for x in dev))
        elif k % 500 == 0:
            print(f"  {k}/{len(names)}")
    if n_fail:
        print(f"[WARN] {n_fail} designs could not be analysed (missing file or chain/CA count mismatch)")
    if not out_rows:
        sys.exit("nothing analysed")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[*] wrote {args.out} ({len(out_rows)} designs)")
    if args.design:
        return

    cd = np.array([o["rmsd_colabdesign"] for o in out_rows])
    print("\nWhich quantity is ColabDesign's `rmsd`?  (median |difference| in A, Pearson r)")
    for key, label in (("bind_tgtalign_mean", "binder after target fit, mean over models"),
                       ("complex_aligned_mean", "whole complex after full fit"),
                       ("complex_raw_mean", "whole complex, no fit")):
        x = np.array([o[key] for o in out_rows])
        ok = np.isfinite(cd) & np.isfinite(x)
        r = np.corrcoef(cd[ok], x[ok])[0, 1] if ok.sum() > 2 else float("nan")
        print(f"  {label:45s} median|diff| {np.median(np.abs(cd[ok] - x[ok])):8.3f}   r = {r:.3f}")

    ip = np.array([o["ipae"] for o in out_rows])
    bt = np.array([o["bind_tgtalign_mean"] for o in out_rows])
    bm = np.array([o["bind_tgtalign_min"] for o in out_rows])
    print(f"\nGate check (iPAE < {args.ipae_cutoff}, RMSD < {args.rmsd_cutoff} A):")
    print(f"  iPAE < cutoff                                   : {int((ip < args.ipae_cutoff).sum())}")
    print(f"  ... and ColabDesign rmsd < cutoff               : {int(((ip < args.ipae_cutoff) & (cd < args.rmsd_cutoff)).sum())}")
    print(f"  ... and binder RMSD after target fit (mean)     : {int(((ip < args.ipae_cutoff) & (bt < args.rmsd_cutoff)).sum())}")
    print(f"  ... and binder RMSD after target fit (best model): {int(((ip < args.ipae_cutoff) & (bm < args.rmsd_cutoff)).sum())}")
    print(f"  lowest binder RMSD after target fit (mean/best) : {np.nanmin(bt):.2f} / {np.nanmin(bm):.2f} A")
    bi = np.array([o["bind_ifacealign_mean"] for o in out_rows])
    tf = np.array([o["target_after_tgtfit_mean"] for o in out_rows])
    itf = np.array([o["iface_fit_rmsd_mean"] for o in out_rows])
    q = lambda x: " / ".join(f"{v:.2f}" for v in np.nanpercentile(x, [5, 50, 95]))
    print(f"\nTarget self-consistency (5/50/95 percentile, A): whole target after fit {q(tf)}; "
          f"interface residues after interface fit {q(itf)}")
    print(f"\nInterface-aligned binder RMSD (fit on target residues within {args.interface_cutoff} A of the binder):")
    print(f"  percentiles 0/5/50 of binder RMSD: {np.nanmin(bi):.2f} / {np.nanpercentile(bi, 5):.2f} / "
          f"{np.nanmedian(bi):.2f} A (target-fit version: {np.nanmin(bt):.2f} / "
          f"{np.nanpercentile(bt, 5):.2f} / {np.nanmedian(bt):.2f} A)")
    ipa = np.array([o["ipae"] for o in out_rows])
    for cut in (0.3, 0.4, 0.5):
        counts = [int(((ipa < cut) & (bi < r)).sum()) for r in (1.5, 2, 3, 4)]
        print(f"  iPAE<{cut}: designs with interface-aligned RMSD <1.5,2,3,4 A = {counts}")
    rc = np.array([o["iface_recall_mean"] for o in out_rows])
    print(f"\nSite recovery (fraction of the designed interface still contacted, {args.contact_cutoff} A CA cutoff):")
    print(f"  all designs, percentiles 5/50/95: {q(rc)}")
    for lo, hi, label in ((0.0, 0.1, "< 0.1  (binder predicted elsewhere)"),
                          (0.1, 0.5, "0.1-0.5 (partial overlap)"),
                          (0.5, 1.01, ">= 0.5  (same site)")):
        sel = (rc >= lo) & (rc < hi)
        print(f"  recall {label:36s}: {int(sel.sum()):6d} designs"
              f"  | of those iPAE<0.4: {int((sel & (ipa < 0.4)).sum())}"
              f"  | RMSD<4 A: {int((sel & (bi < 4)).sum())}")
    if des_cnt is not None and n_hist:
        with open(args.contact_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["target_index", "resid", "designed_frac", "predicted_frac",
                        f"predicted_frac_ipae_lt_{args.ipae_cutoff_hist}"])
            for i in range(len(des_cnt)):
                w.writerow([i + 1, resids[i], f"{des_cnt[i] / n_hist:.4f}", f"{pred_cnt[i] / n_hist:.4f}",
                            f"{pred_cnt_conf[i] / max(n_hist_conf, 1):.4f}"])
        print(f"\nPer-residue contact frequency written to {args.contact_out} "
              f"({n_hist} designs; {n_hist_conf} with iPAE<{args.ipae_cutoff_hist})")
        top = lambda v, n: ", ".join(f"{resids[i]}:{v[i]:.2f}" for i in np.argsort(-v)[:n])
        print(f"  most often in the DESIGNED interface (resid:fraction): {top(des_cnt / n_hist, 10)}")
        print(f"  most often contacted in the PREDICTION               : {top(pred_cnt / n_hist, 10)}")
        print(f"  most contacted by confident (iPAE<{args.ipae_cutoff_hist}) predictions: "
              f"{top(pred_cnt_conf / max(n_hist_conf, 1), 10)}")
    print("\n10 lowest-iPAE designs:")
    print(f"  {'design':36s} {'ipae':>6s} {'colab':>7s} {'bind_tgtalign':>14s} {'per model':>14s} {'recall':>7s}")
    for o in sorted(out_rows, key=lambda o: o["ipae"])[:10]:
        print(f"  {o['design']:36s} {o['ipae']:6.3f} {o['rmsd_colabdesign']:7.2f} "
              f"{o['bind_tgtalign_mean']:14.2f} {o['bind_tgtalign_per_model']:>14s} {o['iface_recall_mean']:7.2f}")


if __name__ == "__main__":
    main()