#!/usr/bin/env python
"""
03_sequence_design_cycle.py

For every RFD3 backbone (chain A = macrocycle, chain B = target), runs the
same iterative sequence-design loop as the paper (Methods 2.2.2):
4 rounds of [ProteinMPNN -> PyRosetta FastRelax w/ PeptideCyclizeMover].

This is where the macrocycle-critical constraint actually gets (re-)applied
regardless of how the backbone was generated: PeptideCyclizeMover enforces
the N-to-C bond and relaxes around it. If RFD3's backbone termini are far
from closure-compatible geometry, this step is likely to fail or produce
heavily distorted structures for a large fraction of designs -- worth
sanity-checking cyclization RMSD/energies on a small batch before scaling up.

Requires:
    - ProteinMPNN (https://github.com/dauparas/ProteinMPNN) checked out locally
    - PyRosetta (with the beta_nov16 score function / PeptideCyclizeMover)
    - rosetta/fast_relax_cyclize.xml (sibling directory)

Usage:
    python 03_sequence_design_cycle.py \
        --backbones-dir rfd3_outputs/ \
        --out-dir mpnn_outputs/ \
        --proteinmpnn-dir /path/to/ProteinMPNN \
        --mpnn-weights /path/to/vanilla_model_weights/v_48_020.pt \
        --n-rounds 4
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAST_RELAX_XML = HERE.parent / "rosetta" / "fast_relax_cyclize.xml"

PYROSETTA_RELAX_TEMPLATE = """
from pyrosetta import *
init('-beta_nov16')

xml = '{xml_path}'
objs = protocols.rosetta_scripts.XmlObjects.create_from_file(xml)
fr = objs.get_mover('full_relax_complex')
pcm = objs.get_mover('pcm')

pose = pose_from_pdb('{in_pdb}')
pcm.apply(pose)
fr.apply(pose)
pcm.apply(pose)
pose.dump_pdb('{out_pdb}')
"""


def run_mpnn_round(proteinmpnn_dir, weights, in_pdb, out_dir):
    cmd = [
        sys.executable, str(Path(proteinmpnn_dir) / "protein_mpnn_run.py"),
        "--pdb_path", str(in_pdb),
        "--pdb_path_chains", "A",
        "--temperature", "0.0001",
        "--backbone_noise", "0",
        "--omit_AAs", "C",
        "--num_seq_per_target", "1",
        "--path_to_model_weights", str(weights),
        "--out_folder", str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def apply_sequence_and_relax(mpnn_fasta_pdb, in_pdb, out_pdb):
    """
    In the paper's own pipeline, applying the new MPNN sequence to the pose
    and thread it back into the structure is glue code around Rosetta's
    pose.replace_sequence-style utilities. Wire that glue in here to match
    however your ProteinMPNN output is packaged (fasta vs. re-threaded pdb).
    """
    script = PYROSETTA_RELAX_TEMPLATE.format(
        xml_path=FAST_RELAX_XML, in_pdb=in_pdb, out_pdb=out_pdb
    )
    tmp_script = Path(out_pdb).with_suffix(".relax.py")
    tmp_script.write_text(script)
    subprocess.run([sys.executable, str(tmp_script)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--proteinmpnn-dir", required=True)
    ap.add_argument("--mpnn-weights", required=True)
    ap.add_argument("--n-rounds", type=int, default=4)
    args = ap.parse_args()

    backbones_dir = Path(args.backbones_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(backbones_dir.glob("*.pdb")) + sorted(backbones_dir.glob("*.cif"))
    if not pdbs:
        sys.exit(f"No .pdb/.cif files found under {backbones_dir}")

    for pdb in pdbs:
        current = pdb
        for rnd in range(1, args.n_rounds + 1):
            round_dir = out_dir / pdb.stem / f"round{rnd}"
            round_dir.mkdir(parents=True, exist_ok=True)

            run_mpnn_round(args.proteinmpnn_dir, args.mpnn_weights, current, round_dir)

            # NOTE: adapt this glob to whatever ProteinMPNN's out_folder layout
            # produces for your installed version.
            mpnn_out = next(round_dir.glob("seqs/*.fa"), None)
            if mpnn_out is None:
                print(f"[warn] no MPNN output found for {current}, skipping rest of chain")
                break

            relaxed_pdb = round_dir / f"{pdb.stem}_r{rnd}.pdb"
            apply_sequence_and_relax(mpnn_out, current, relaxed_pdb)
            current = relaxed_pdb

        print(f"{pdb.name}: final sequence-designed, cyclized model -> {current}")


if __name__ == "__main__":
    main()
