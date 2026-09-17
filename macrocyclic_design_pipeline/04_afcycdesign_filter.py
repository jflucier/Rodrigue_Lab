#!/usr/bin/env python
"""
04_afcycdesign_filter.py

Predicts each sequence-designed, cyclized macrocycle-target complex with
AfCycDesign (ColabDesign fork with cyclic offset, Rettie et al. 2025
Methods 2.3) and filters on normalized iPAE (iPAE/32) and Ca RMSD to the
design model, matching the paper's per-target cutoffs:

    MCL1:     normalized iPAE < 0.20
    MDM2:     normalized iPAE < 0.30
    GABARAP:  normalized iPAE < 0.13
    RbtA:     normalized iPAE < 0.40   (+ Ca RMSD < 1.5 A used alongside iPAE)

These cutoffs are target/system-specific (the paper picked them per-target
from the observed iPAE distribution, see Methods 2.3) -- treat the defaults
below as starting points, not universal constants, and re-derive them from
your own iPAE histograms per target.

Requires ColabDesign installed with AfCycDesign's add_cyclic_offset patch:
https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/af/examples/af_cyc_design.ipynb

Usage:
    python 04_afcycdesign_filter.py \
        --designs-dir mpnn_outputs/ \
        --out-csv afcyc_scores.csv \
        --norm-ipae-cutoff 0.20 \
        --rmsd-cutoff 1.5
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    from colabdesign.af import mk_afdesign_model
    from colabdesign.af.alphafold.common import residue_constants  # noqa: F401
except ImportError:
    mk_afdesign_model = None  # allow --dry-run without ColabDesign installed


def add_cyclic_offset(model, offset_type=2):
    """
    Placeholder hook: import/apply AfCycDesign's actual add_cyclic_offset
    implementation here (from the af_cyc_design ColabDesign notebook linked
    above). Left as a stub so this script doesn't silently vendor someone
    else's function incorrectly -- copy it in from the notebook you're
    running against, matching whatever ColabDesign version you have pinned.
    """
    raise NotImplementedError(
        "Wire in AfCycDesign's add_cyclic_offset from the ColabDesign "
        "af_cyc_design notebook before running for real."
    )


def predict_one(pdb_path: Path, out_dir: Path):
    model = mk_afdesign_model("binder")
    model.prep_inputs(
        str(pdb_path),
        binder_chain="B",
        target_chain="A",
        use_binder_template=False,
        use_multimer=True,
        use_initial_guess=True,
    )
    add_cyclic_offset(model, offset_type=2)
    model.set_seq(mode="wildtype")
    model.set_opt(num_recycles=1)
    model.predict(models=[0, 1], verbose=False)

    out_pdb = out_dir / f"{pdb_path.stem}_prediction.pdb"
    model.save_pdb(str(out_pdb))

    rmsd = model.aux["losses"]["rmsd"]
    ipae = model.aux["all"]["losses"]["i_pae"][0]
    return rmsd, ipae


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs-dir", required=True,
                     help="Directory of cyclized, sequence-designed complex PDBs "
                          "(chain A = target per paper's convention here, chain B = macrocycle "
                          "-- adjust to match script 03's output chain order)")
    ap.add_argument("--out-csv", default="afcyc_scores.csv")
    ap.add_argument("--norm-ipae-cutoff", type=float, default=0.20)
    ap.add_argument("--rmsd-cutoff", type=float, default=1.5)
    ap.add_argument("--ipae-norm", type=float, default=32.0,
                     help="Divisor used to normalize raw iPAE (paper: iPAE/32)")
    ap.add_argument("--dry-run", action="store_true",
                     help="List inputs and exit without running ColabDesign")
    args = ap.parse_args()

    designs_dir = Path(args.designs_dir)
    pdbs = sorted(designs_dir.rglob("*.pdb"))
    if not pdbs:
        sys.exit(f"No PDBs found under {designs_dir}")

    if args.dry_run:
        print(f"Would score {len(pdbs)} structures.")
        return

    if mk_afdesign_model is None:
        sys.exit("ColabDesign not importable -- install it or use --dry-run.")

    pred_dir = designs_dir.parent / "afcyc_predictions"
    pred_dir.mkdir(exist_ok=True)

    rows = []
    for pdb in pdbs:
        try:
            rmsd, ipae = predict_one(pdb, pred_dir)
        except Exception as e:
            print(f"[warn] {pdb.name}: prediction failed ({e})")
            continue
        norm_ipae = ipae / args.ipae_norm
        passed = norm_ipae < args.norm_ipae_cutoff and rmsd < args.rmsd_cutoff
        rows.append({
            "design": pdb.stem, "rmsd": rmsd, "ipae": ipae,
            "normalized_ipae": norm_ipae, "passed": passed,
        })

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["design", "rmsd", "ipae",
                                                  "normalized_ipae", "passed"])
        writer.writeheader()
        writer.writerows(rows)

    n_passed = sum(r["passed"] for r in rows)
    print(f"{n_passed}/{len(rows)} designs passed "
          f"(normalized iPAE < {args.norm_ipae_cutoff}, RMSD < {args.rmsd_cutoff} A)")
    print(f"Scores written to {args.out_csv}")


if __name__ == "__main__":
    main()
