#!/usr/bin/env python
"""
00_preprocess_pdb.py

Step 0 of the macrocycle-binder pipeline. Per the paper's Supplementary
Methods (2.2.1): "the PDB files corresponding to the selected target were
downloaded from the Protein Data Bank and stripped of all water and
ligands, leaving just the target protein atoms."

Reads the same targets TSV used by 01_build_rfpeptides_runs.py
(design_name, pdb_path, hotspots[, length]), strips waters and
non-polymer/heteroatom records (crystallization additives, ions, small-
molecule ligands, etc.) from each PDB, keeping only standard protein
residues, and writes cleaned structures to 00_pdb_processed/.

Also writes an updated TSV (targets_processed.tsv, alongside the output
dir) with pdb_path repointed at the cleaned files, so it can be handed
straight to 01_build_rfpeptides_runs.py.

Intended location on your system:
    /storage/Documents/service/biologie/rodrigue/programs/rodrigue_lab/macrocyclic_design_pipeline/scripts/00_preprocess_pdb.py
Run with --out-dir pointed at your per-analysis working directory (output
PDBs land in ./00_pdb_processed/ relative to wherever you run from, by
default).

Usage:
    python 00_preprocess_pdb.py configs/targets.tsv \
        --out-dir 00_pdb_processed \
        --out-tsv targets_processed.tsv

Notes:
- "Ligands" here means anything not a standard amino acid polymer residue:
  HETATM records for waters, ions, buffer/cryo additives, and any bound
  small-molecule/peptide ligands are all removed, matching "leaving just
  the target protein atoms." If a target's crystallized WITH something you
  actually want to keep (e.g. a cofactor relevant to the pocket), pass
  --keep-hetero <RES_NAME> one or more times to whitelist it explicitly --
  nothing is kept by accident otherwise.
- Only the first model of multi-model files (e.g. NMR ensembles) is kept.
- Multi-letter/insertion-code residues and alternate locations are handled
  by biotite's default parsing (first altloc kept).
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.pdbx as pdbx_io
except ImportError:
    sys.exit("This script needs biotite: pip install biotite")


def load_structure(pdb_path: Path):
    if pdb_path.suffix.lower() in (".cif", ".mmcif"):
        cif = pdbx_io.CIFFile.read(str(pdb_path))
        return pdbx_io.get_structure(cif, model=1), "pdb"  # write out as .pdb regardless
    f = pdb_io.PDBFile.read(str(pdb_path))
    return pdb_io.get_structure(f, model=1), "pdb"


def strip_water_and_ligands(structure, keep_hetero=None):
    keep_hetero = set(keep_hetero or [])

    # struc.filter_amino_acids selects standard protein residues (by res_name,
    # backed by biotite's residue-name table) -- this is the core "just the
    # target protein atoms" filter.
    is_protein = struc.filter_amino_acids(structure)

    if keep_hetero:
        is_whitelisted = (
            struc.filter_hetero(structure) & (
                structure.res_name[..., None] == list(keep_hetero)
            ).any(axis=-1)
        )
        mask = is_protein | is_whitelisted
    else:
        mask = is_protein

    n_removed = (~mask).sum()
    return structure[mask], int(n_removed)


def write_pdb(structure, out_path: Path):
    out_file = pdb_io.PDBFile()
    pdb_io.set_structure(out_file, structure)
    out_file.write(str(out_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets_tsv", help="TSV: design_name, pdb_path, hotspots[, length]")
    ap.add_argument("--out-dir", default="00_pdb_processed",
                     help="Directory to write cleaned PDBs into")
    ap.add_argument("--out-tsv", default=None,
                     help="Path for the rewritten TSV pointing at cleaned PDBs "
                          "(default: targets_processed.tsv next to --out-dir)")
    ap.add_argument("--keep-hetero", action="append", default=[],
                     metavar="RES_NAME",
                     help="Heteroatom residue name to keep instead of stripping "
                          "(e.g. a cofactor). Repeatable. Nothing is kept unless "
                          "explicitly whitelisted here.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = Path(args.out_tsv) if args.out_tsv else out_dir.parent / "targets_processed.tsv"

    with open(args.targets_tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"design_name", "pdb_path", "hotspots"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"TSV missing required columns: {missing}")
        fieldnames = reader.fieldnames
        rows = list(reader)

    out_rows = []
    for row in rows:
        name = row["design_name"].strip()
        if not name:
            continue
        in_path = Path(row["pdb_path"].strip())
        if not in_path.exists():
            sys.exit(f"[{name}] pdb_path does not exist: {in_path}")

        structure, _ = load_structure(in_path)
        n_atoms_before = structure.array_length()
        cleaned, n_removed = strip_water_and_ligands(structure, args.keep_hetero)

        out_path = out_dir / f"{name}.pdb"
        write_pdb(cleaned, out_path)

        print(f"[{name}] {in_path.name}: removed {n_removed}/{n_atoms_before} "
              f"non-protein atoms -> {out_path}")

        new_row = dict(row)
        new_row["pdb_path"] = str(out_path)
        out_rows.append(new_row)

    with open(out_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} cleaned PDB(s) to {out_dir}/")
    print(f"Wrote updated TSV to {out_tsv} (pass this to 01_build_rfpeptides_runs.py)")


if __name__ == "__main__":
    main()
