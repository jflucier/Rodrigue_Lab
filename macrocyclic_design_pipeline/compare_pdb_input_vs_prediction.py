#!/usr/bin/env python3
"""
compare_input_vs_prediction.py

Compares an AfCycDesign prediction with its input complex (Calpha atoms only):
  - target RMSD after superposing the prediction onto the input by the target
  - binder RMSD after that same target superposition (what ColabDesign calls rmsd)
  - residues that drift most in the target (flexible tails vs. real template loss)

ColabDesign writes the target as chain A and the binder as chain B in its
prediction PDB, so the --pred-* defaults are A and B. Set --in-* to match the
input file (e.g. 1YCR: --in-target A --in-binder B).
"""
import argparse
import numpy as np


def ca(path, chain):
    xyz = [[float(l[30:38]), float(l[38:46]), float(l[46:54])]
           for l in open(path)
           if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[21] == chain]
    if not xyz:
        raise SystemExit(f"No CA atoms for chain {chain} in {path}")
    return np.array(xyz)


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T, pc, qc


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("input_pdb")
ap.add_argument("prediction_pdb")
ap.add_argument("--in-target", required=True)
ap.add_argument("--in-binder", required=True)
ap.add_argument("--pred-target", default="A")
ap.add_argument("--pred-binder", default="B")
a = ap.parse_args()

tI, bI = ca(a.input_pdb, a.in_target), ca(a.input_pdb, a.in_binder)
tP, bP = ca(a.prediction_pdb, a.pred_target), ca(a.prediction_pdb, a.pred_binder)
if len(tI) != len(tP) or len(bI) != len(bP):
    raise SystemExit(f"Length mismatch: target {len(tI)} vs {len(tP)}, binder {len(bI)} vs {len(bP)}")

R, pc, qc = kabsch(tP, tI)
move = lambda x: (x - pc) @ R.T + qc
dev = np.linalg.norm(move(tP) - tI, axis=1)
print(f"target residues {len(tI)}, binder residues {len(bI)}")
print(f"target RMSD (all)            : {rmsd(move(tP), tI):.2f} A")
print(f"binder RMSD (target-aligned) : {rmsd(move(bP), bI):.2f} A")

mask = np.ones(len(tI), bool)
for _ in range(10):
    R3, p3, q3 = kabsch(tP[mask], tI[mask])
    d = np.linalg.norm((tP - p3) @ R3.T + q3 - tI, axis=1)
    mask = d < 2.0
print(f"target core (<2 A, iterative): {mask.sum()}/{len(tI)} residues, core RMSD {np.sqrt((d[mask] ** 2).mean()):.2f} A")
top = np.argsort(dev)[::-1][:8]
print("largest target deviations (residue index:A): " + " ".join(f"{i + 1}:{dev[i]:.1f}" for i in sorted(top)))
dI = np.linalg.norm(bI[:, None] - tI[None], axis=-1)
dP = np.linalg.norm(bP[:, None] - tP[None], axis=-1)
print(f"binder-target contacts <8 A  : input {(dI < 8).sum()}, prediction {(dP < 8).sum()}")
