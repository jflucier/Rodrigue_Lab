#!/usr/bin/env python
"""
02b_normalize_chains.py

RFdiffusion/RFpeptides does NOT reliably put the diffused macrocycle on
chain A. It honors whatever chain letter the contig gives the FIXED
(target) block, and assigns the unlabeled diffused block whatever letter is
left over -- so if your target's original chain happens to be "A" (as in
contigmap.contigs=[8-18 A1-77/0]), the macrocycle often comes out as chain
B and the target keeps chain A. Every downstream script/XML in this
pipeline (fast_relax_cyclize.xml, interface_metrics.xml,
03b_fastrelax_worker.py, the ProteinMPNN --pdb_path_chains flag) assumes
macrocycle=chain A, target=chain B. Run this step between backbone
generation (02_run_rfpeptides.sh) and sequence design (03_...) to make that
assumption true for every backbone, regardless of what RFdiffusion actually
emitted.

Heuristic: the macrocycle is always much shorter than the target domains in
this pipeline (8-18 residues vs. 70-300+), so the chain with FEWER residues
is relabeled 'A' (macrocycle) and all others get 'B', 'C', ... in their
original relative order. A file where this heuristic looks unsafe (fewer
than 2 chains, or the "shorter" chain is already implausibly long) is
skipped with a loud warning rather than silently mislabeled -- check those
by hand.

Usage:
    python 02b_normalize_chains.py rfpeptides_runs/dnaJ_campaign \
        --out-dir rfpeptides_runs_normalized/dnaJ_campaign \
        --max-macrocycle-len 30
"""
import argparse
import shutil
import sys
from pathlib import Path

try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io
except ImportError:
    sys.exit("This script needs biotite: pip install biotite")


def normalize_one(structure, max_macrocycle_len):
    chains = list(dict.fromkeys(structure.chain_id))  # unique, order-preserving
    if len(chains) < 2:
        return None, f"only {len(chains)} chain(s), nothing to normalize"

    counts = {}
    for c in chains:
        mask = structure.chain_id == c
        counts[c] = len(set(zip(structure.res_id[mask], structure.ins_code[mask])))

    macrocycle_chain = min(counts, key=counts.get)
    if counts[macrocycle_chain] > max_macrocycle_len:
        return None, (f"shortest chain ({macrocycle_chain}, {counts[macrocycle_chain]} res) "
                       f"exceeds --max-macrocycle-len={max_macrocycle_len} -- "
                       f"refusing to guess which chain is the macrocycle "
                       f"(chain sizes: {counts})")

    other_chains = [c for c in chains if c != macrocycle_chain]
    new_letters = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(other_chains) > len(new_letters):
        return None, f"too many non-macrocycle chains ({len(other_chains)}) to relabel"

    remap = {macrocycle_chain: "A"}
    remap.update({c: new_letters[i] for i, c in enumerate(other_chains)})

    new_chain_id = structure.chain_id.copy()
    for old, new in remap.items():
        new_chain_id[structure.chain_id == old] = new
    structure.chain_id = new_chain_id

    note = f"chain {macrocycle_chain} ({counts[macrocycle_chain]} res) -> A (macrocycle); " + \
           ", ".join(f"{old} -> {new}" for old, new in remap.items() if old != macrocycle_chain)
    return structure, note


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backbones_dir")
    ap.add_argument("--out-dir", required=True,
                     help="Where normalized copies are written (mirrors input filenames)")
    ap.add_argument("--max-macrocycle-len", type=int, default=18,
                     help="Sanity bound: if the shortest chain exceeds this many "
                          "residues, the file is skipped instead of guessed at")
    args = ap.parse_args()

    in_dir = Path(args.backbones_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(in_dir.glob("*.pdb"))
    if not pdbs:
        sys.exit(f"No .pdb files found under {in_dir}")

    n_ok, n_skipped = 0, 0
    for pdb in pdbs:
        f = pdb_io.PDBFile.read(str(pdb))
        structure = pdb_io.get_structure(f, model=1)
        normalized, note = normalize_one(structure, args.max_macrocycle_len)

        if normalized is None:
            print(f"[SKIP] {pdb.name}: {note}")
            n_skipped += 1
            continue

        out_file = pdb_io.PDBFile()
        pdb_io.set_structure(out_file, normalized)
        out_file.write(str(out_dir / pdb.name))
        print(f"[OK]   {pdb.name}: {note}")
        n_ok += 1

    print(f"\n{n_ok} normalized, {n_skipped} skipped -> {out_dir}")
    if n_skipped:
        print("Skipped files need manual inspection before running sequence design on them.")


if __name__ == "__main__":
    main()
