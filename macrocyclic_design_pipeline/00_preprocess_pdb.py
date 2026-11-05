#!/usr/bin/env python
"""
00_preprocess_pdb.py

Step 0 of the macrocycle-binder pipeline. Per the paper's Supplementary
Methods (2.2.1): "the PDB files corresponding to the selected target were
downloaded from the Protein Data Bank and stripped of all water and
ligands, leaving just the target protein atoms."

Reads the same targets TSV used by 01_build_rfpeptides_runs.py
(design_name, pdb_path[, target_chain][, hotspots][, length][, target_pos]),
strips waters and
non-polymer/heteroatom records (crystallization additives, ions, small-
molecule ligands, etc.) from each PDB, keeping only standard protein
residues, and writes cleaned structures to 00_pdb_processed/.

Also writes an updated TSV (targets_processed.tsv, alongside the output
dir) with pdb_path repointed at the cleaned files, so it can be handed
straight to 01_build_rfpeptides_runs.py.

Column target_chain: the chain of the target to design against ("A"). Needed
when a row has no hotspots (unless the PDB has a single chain); optional
otherwise, but must match the hotspot chain. Column hotspots may be left empty
to design without hotspot guidance. Both columns are always written to
targets_processed.tsv (the resolved chain is filled in).

Only the target chain is written to the cleaned PDB (other protein chains are
dropped, original chain ID and numbering kept); use --keep-all-chains to
disable that.

Optional column target_pos: the target residues to design against, as one
contiguous range in the input PDB's own numbering, e.g. "1-66" (a chain
letter prefix such as "A1-66" is also accepted; it must match the hotspot
chain). Empty/absent = whole target chain. The cleaned PDB is NOT cropped:
cropping is applied by 01_build_rfpeptides_runs.py through the RFdiffusion
contig, so hotspot numbering is unchanged. This script always writes the
target_pos column to targets_processed.tsv (blank when not provided) and
checks that the range lies inside the chain and contains every hotspot.

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
import re
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


def parse_hotspots(hotspot_str: str):
    """Return the list of hotspot tokens (e.g. ['A68','A70']); empty list = no hotspots."""
    return [t.strip() for t in (hotspot_str or "").split(",") if t.strip()]


def resolve_chain(structure, hotspot_tokens, target_chain: str):
    """Decide which chain is the target.

    - hotspots given: their chain is the target chain; if target_chain is also
      given it must agree.
    - no hotspots: target_chain is used; if it is blank and the structure has a
      single chain, that chain is used; otherwise it is an error.
    """
    hs_chains = {t[0] for t in hotspot_tokens}
    if len(hs_chains) > 1:
        raise ValueError(
            f"All hotspots in one row must be on the same chain, got: {','.join(hotspot_tokens)}")
    target_chain = (target_chain or "").strip()
    if hs_chains:
        chain = next(iter(hs_chains))
        if target_chain and target_chain != chain:
            raise ValueError(
                f"target_chain {target_chain!r} differs from hotspot chain {chain!r}")
        return chain
    if target_chain:
        return target_chain
    chains = sorted(set(structure.chain_id.tolist()))
    if len(chains) == 1:
        return chains[0]
    raise ValueError(
        "no hotspots and no target_chain given, and the structure has several chains "
        f"({', '.join(chains)}); set the target_chain column")


def _ranges(nums):
    """[1,2,3,7,9,10] -> [(1,3),(7,7),(9,10)]"""
    out = []
    for n in sorted(nums):
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def _fmt(ranges):
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)


def describe_gaps(structure, chain: str, lo: int, hi: int, original=None) -> str:
    """RFdiffusion needs EVERY residue number in the contig range (chain+lo-hi) to
    exist in the PDB ("('A', 198) is not in pdb file!" otherwise). Returns a
    readable message if the range has holes, else "". If `original` (the PDB before
    cleaning) is given, also reports holes caused by our own stripping."""
    present = {int(r) for r in structure.res_id[structure.chain_id == chain]}
    missing = [i for i in range(lo, hi + 1) if i not in present]
    if not missing:
        return ""
    segs = _ranges([i for i in range(lo, hi + 1) if i in present])
    msg = (f"target chain {chain} residues {lo}-{hi}: {len(missing)} residue number(s) "
           f"are not in the PDB ({_fmt(_ranges(missing))}), but RFdiffusion needs the "
           f"whole contig range to exist. Contiguous stretches present: "
           + ", ".join(f"{a}-{b} ({b - a + 1} res)" for a, b in segs)
           + ". Set target_pos in the TSV to one contiguous stretch (it must contain "
           "all hotspots).")
    if original is not None:
        om = original.chain_id == chain
        names = {}
        for rid, rn in zip(original.res_id[om], original.res_name[om]):
            names.setdefault(int(rid), str(rn))
        stripped = {i: names[i] for i in missing if i in names}
        if stripped:
            kinds = sorted(set(stripped.values()))
            msg += (f" NOTE: {len(stripped)} of the missing numbers ({_fmt(_ranges(stripped))}) "
                    f"exist in the input PDB but were removed as non-protein "
                    f"({', '.join(kinds)}); if these are modified residues you want, "
                    f"re-run with --keep-hetero <RES_NAME>.")
    return msg


def check_target_pos(structure, tokens, chain: str, target_pos: str):
    """Validate target_pos against the cleaned structure and the hotspots (if any).
    Returns (chain, lo, hi, full_lo, full_hi)."""

    m = re.fullmatch(r"([A-Za-z]?)(\d+)\s*-\s*(\d+)", target_pos.strip())
    if not m:
        raise ValueError(f"target_pos {target_pos!r} must be one range like '1-66' or 'A1-66'")
    tp_chain, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
    if tp_chain and tp_chain != chain:
        raise ValueError(f"target_pos chain {tp_chain} differs from hotspot chain {chain}")
    if lo > hi:
        raise ValueError(f"target_pos {lo}-{hi} is reversed")

    mask = structure.chain_id == chain
    if not mask.any():
        raise ValueError(f"chain {chain} not found in cleaned structure")
    full_lo, full_hi = int(structure.res_id[mask].min()), int(structure.res_id[mask].max())
    if lo < full_lo or hi > full_hi:
        raise ValueError(f"target_pos {lo}-{hi} is outside chain {chain} residues {full_lo}-{full_hi}")

    try:
        positions = [int(t[1:]) for t in tokens]
    except ValueError:
        raise ValueError(f"cannot parse hotspot residue numbers in {tokens}")
    outside = [t for t, n in zip(tokens, positions) if not lo <= n <= hi]
    if outside:
        raise ValueError(f"hotspots outside target_pos {lo}-{hi}: {', '.join(outside)}")
    return chain, lo, hi, full_lo, full_hi


def write_pdb(structure, out_path: Path):
    out_file = pdb_io.PDBFile()
    pdb_io.set_structure(out_file, structure)
    out_file.write(str(out_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets_tsv",
                     help="TSV: design_name, pdb_path[, target_chain][, hotspots][, length][, target_pos]")
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
    ap.add_argument("--keep-all-chains", action="store_true",
                     help="Keep every protein chain instead of only the target chain "
                          "(default: only the target chain is written).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()  # Convert output directory to absolute path
    out_dir.mkdir(parents=True, exist_ok=True)

    out_tsv = Path(args.out_tsv) if args.out_tsv else out_dir.parent / "targets_processed.tsv"
    out_tsv = out_tsv.resolve()  # Convert output TSV to absolute path


    with open(args.targets_tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"design_name", "pdb_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"TSV missing required columns: {missing}")
        fieldnames = list(reader.fieldnames)
        for col in ("hotspots", "target_chain", "target_pos"):
            if col not in fieldnames:
                fieldnames.append(col)  # always emitted, blank if not provided
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
        print(f"[{name}] {in_path.name}: removed {n_removed}/{n_atoms_before} "
              f"non-protein atoms")

        hotspot_tokens = parse_hotspots(row.get("hotspots"))
        try:
            chain = resolve_chain(cleaned, hotspot_tokens, row.get("target_chain"))
        except ValueError as e:
            sys.exit(f"[{name}] {e}")
        if not (cleaned.chain_id == chain).any():
            sys.exit(f"[{name}] chain {chain} not found in cleaned structure")

        # Keep only the target chain (original chain ID and residue numbering are
        # untouched, so hotspots and target_pos still refer to the same residues).
        if not args.keep_all_chains:
            other = sorted(set(cleaned.chain_id.tolist()) - {chain})
            cleaned = cleaned[cleaned.chain_id == chain]
            if other:
                print(f"[{name}] kept target chain {chain} only; dropped chain(s): "
                      f"{', '.join(other)}")

        target_pos = (row.get("target_pos") or "").strip()
        if target_pos:
            try:
                chain, lo, hi, full_lo, full_hi = check_target_pos(
                    cleaned, hotspot_tokens, chain, target_pos)
            except ValueError as e:
                sys.exit(f"[{name}] {e}")
            print(f"[{name}] target_pos {target_pos}: chain {chain} residues {lo}-{hi} "
                  f"({hi - lo + 1} of {full_hi - full_lo + 1}); "
                  + ("hotspots inside range" if hotspot_tokens else "no hotspots"))
        else:
            print(f"[{name}] no target_pos: whole target will be used")
            lo = full_lo = int(cleaned.res_id[cleaned.chain_id == chain].min())
            hi = full_hi = int(cleaned.res_id[cleaned.chain_id == chain].max())

        gap_msg = describe_gaps(cleaned, chain, lo, hi, original=structure)
        if gap_msg:
            sys.exit(f"[{name}] {gap_msg}")

        write_pdb(cleaned, out_path)
        print(f"[{name}] wrote {out_path}")

        new_row = dict(row)
        new_row["pdb_path"] = str(out_path)
        new_row["target_pos"] = target_pos
        new_row["target_chain"] = chain
        out_rows.append(new_row)

    with open(out_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} cleaned PDB(s) to {out_dir}/")
    print(f"Wrote updated TSV to {out_tsv} (pass this to 01_build_rfpeptides_runs.py)")


if __name__ == "__main__":
    main()
