#!/usr/bin/env python
"""
05_rosetta_filter.py

Runs rosetta/interface_metrics.xml (ddG, SAP, contact molecular surface
area) on every AfCycDesign-passing complex and applies target-specific
cutoffs, mirroring the paper's per-target filters (Methods/Supp. Table
captions):

    MCL1:    ddG < -40 kcal/mol, SAP < 35, CMS > 300 A^2
    MDM2:    ddG < -50 kcal/mol, SAP < 35, CMS > 300 A^2
    GABARAP: ddG < -30 kcal/mol, SAP < 35, CMS > 300 A^2
    RbtA:    ddG < -40 kcal/mol, SAP < 35, CMS > 300 A^2

Requires a Rosetta build with rosetta_scripts on PATH (or pass --rosetta-bin).

Usage:
    python 05_rosetta_filter.py \
        --designs-dir afcyc_passing/ \
        --out-csv rosetta_scores.csv \
        --ddg-cutoff -40 --sap-cutoff 35 --cms-cutoff 300
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERFACE_XML = HERE.parent / "rosetta" / "interface_metrics.xml"


def run_rosetta_scripts(rosetta_bin, pdb_path, out_dir):
    cmd = [
        rosetta_bin,
        "-parser:protocol", str(INTERFACE_XML),
        "-in:file:s", str(pdb_path),
        "-out:path:all", str(out_dir),
        "-out:file:scorefile", "score.sc",
        "-overwrite",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def parse_score_file(score_sc: Path, design_name: str):
    """
    Parses Rosetta's whitespace-delimited score.sc for the ddg / sap_score /
    contact_molecular_surface columns for the most recent entry matching
    design_name. Adjust column names here if your Rosetta version reports
    them under slightly different headers.
    """
    lines = score_sc.read_text().splitlines()
    header_line = next((l for l in lines if l.startswith("SCORE:") and "total_score" in l), None)
    if header_line is None:
        raise RuntimeError(f"Could not find score header in {score_sc}")
    columns = header_line.split()[1:]

    matches = [l for l in lines if l.startswith("SCORE:") and design_name in l]
    if not matches:
        raise RuntimeError(f"No score entry for {design_name} in {score_sc}")
    values = matches[-1].split()[1:]
    row = dict(zip(columns, values))

    def find(pattern):
        for k, v in row.items():
            if re.search(pattern, k, re.IGNORECASE):
                return float(v)
        return None

    return {
        "ddg": find(r"ddg"),
        "sap": find(r"sap"),
        "cms": find(r"contact_molecular_surface|cms"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs-dir", required=True)
    ap.add_argument("--out-csv", default="rosetta_scores.csv")
    ap.add_argument("--rosetta-bin", default="rosetta_scripts.linuxgccrelease",
                     help="rosetta_scripts binary name/path. Default matches a "
                          "from-source scons build (see rosetta_pyrosetta_build.def); "
                          "RosettaCommons' prebuilt binary bundles instead use "
                          "rosetta_scripts.default.linuxgccrelease -- adjust if you're "
                          "using those instead.")
    ap.add_argument("--ddg-cutoff", type=float, default=-40.0,
                     help="ddG must be less than this (more negative = better)")
    ap.add_argument("--sap-cutoff", type=float, default=35.0,
                     help="SAP must be less than this")
    ap.add_argument("--cms-cutoff", type=float, default=300.0,
                     help="Contact molecular surface must exceed this")
    args = ap.parse_args()

    designs_dir = Path(args.designs_dir)
    pdbs = sorted(designs_dir.rglob("*.pdb"))
    if not pdbs:
        sys.exit(f"No PDBs found under {designs_dir}")

    work_dir = designs_dir.parent / "rosetta_metrics"
    work_dir.mkdir(exist_ok=True)

    rows = []
    for pdb in pdbs:
        pdb_work_dir = work_dir / pdb.stem
        pdb_work_dir.mkdir(exist_ok=True)
        try:
            run_rosetta_scripts(args.rosetta_bin, pdb, pdb_work_dir)
            metrics = parse_score_file(pdb_work_dir / "score.sc", pdb.stem)
        except Exception as e:
            print(f"[warn] {pdb.name}: rosetta run/parse failed ({e})")
            continue

        passed = (
            metrics["ddg"] is not None and metrics["ddg"] < args.ddg_cutoff
            and metrics["sap"] is not None and metrics["sap"] < args.sap_cutoff
            and metrics["cms"] is not None and metrics["cms"] > args.cms_cutoff
        )
        rows.append({"design": pdb.stem, **metrics, "passed": passed})

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["design", "ddg", "sap", "cms", "passed"])
        writer.writeheader()
        writer.writerows(rows)

    n_passed = sum(r["passed"] for r in rows)
    print(f"{n_passed}/{len(rows)} designs passed "
          f"(ddG < {args.ddg_cutoff}, SAP < {args.sap_cutoff}, CMS > {args.cms_cutoff})")
    print(f"Scores written to {args.out_csv}")


if __name__ == "__main__":
    main()
