#!/usr/bin/env python
"""
03b_fastrelax_worker.py

Runs INSIDE the container (PyRosetta). For each job in a TSV
(in_pdb <TAB> mpnn_fasta <TAB> out_pdb):
  1. read the DESIGNED sequence from the ProteinMPNN fasta
  2. thread it onto chain A (the macrocycle)
  3. PeptideCyclizeMover -> FastRelax -> PeptideCyclizeMover (paper XML)
  4. dump out_pdb

PyRosetta is initialised once and the XML movers are built once per
process, so a whole chunk of backbones is handled per container launch.
A failure on one job is logged and does not stop the rest.
"""
import argparse
import sys
import traceback
from pathlib import Path

from pyrosetta import init, pose_from_pdb
import pyrosetta.rosetta.protocols.rosetta_scripts as rosetta_scripts
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

AA_1TO3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}


def read_designed_sequence(fasta_path):
    """
    ProteinMPNN fasta layout: record 1 = the INPUT (native) sequence,
    records 2.. = designed samples. With --num_seq_per_target 1 we want
    record 2. Chains are separated by '/', chain A (designed) first.
    """
    records, buf, in_rec = [], [], False
    for line in Path(fasta_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_rec:
                records.append("".join(buf))
            buf, in_rec = [], True
        else:
            buf.append(line)
    if in_rec:
        records.append("".join(buf))
    if len(records) < 2:
        raise ValueError(f"{fasta_path}: expected native + >=1 designed record, "
                         f"found {len(records)}")
    return records[1].split("/")[0].upper()


def terminus_gap(pose, chain_a):
    """
    Distance between the chain's C-terminal C atom and N-terminal N atom --
    i.e. the gap the would-be peptide bond has to close. A real peptide
    bond is ~1.33 A; anything much larger means this particular backbone's
    termini were never actually brought together by RFdiffusion sampling,
    and PeptideCyclizeMover/FastRelax failing on it reflects a bad sample,
    not a pipeline bug.
    """
    first_res = pose.residue(chain_a[0])
    last_res = pose.residue(chain_a[-1])
    n_xyz = first_res.xyz("N")
    c_xyz = last_res.xyz("C")
    return (n_xyz - c_xyz).norm()


def run_job(in_pdb, fasta, out_pdb, pcm, fr):
    seq = read_designed_sequence(fasta)
    pose = pose_from_pdb(in_pdb)
    info = pose.pdb_info()
    chain_a = [i for i in range(1, pose.size() + 1) if info.chain(i) == "A"]

    if len(seq) != len(chain_a):
        raise ValueError(f"designed seq length {len(seq)} != chain A length "
                         f"{len(chain_a)} in {in_pdb}")
    bad = sorted(set(seq) - set(AA_1TO3))
    if bad:
        raise ValueError(f"non-standard characters in designed sequence: {bad}")

    gap = terminus_gap(pose, chain_a)
    print(f"  threading {seq} onto chain A ({len(chain_a)} res), "
          f"N-C terminus gap = {gap:.2f} A (expect ~1.3 A for a closed bond)",
          flush=True)
    if gap > 3.0:
        print(f"  [NOTE] gap is large -- this backbone's termini likely never "
              f"closed during RFdiffusion sampling; a cyclization failure below "
              f"is expected for this one, not a pipeline bug", flush=True)

    mutator = MutateResidue()
    for pose_idx, aa in zip(chain_a, seq):
        mutator.set_target(pose_idx)
        mutator.set_res_name(AA_1TO3[aa])
        mutator.apply(pose)

    pcm.apply(pose)
    fr.apply(pose)
    pcm.apply(pose)
    Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
    pose.dump_pdb(out_pdb)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-tsv", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--debug", action="store_true",
                     help="Don't mute Rosetta tracers -- shows PeptideCyclizeMover's "
                          "own diagnostic output for WHY a bond couldn't be "
                          "established, instead of just the later CountPairFactory "
                          "assertion. Much noisier; use on a small --jobs-tsv of "
                          "just the failing backbones, not a full production chunk.")
    args = ap.parse_args()

    init("-beta_nov16 -detect_links false -cyclic_peptide" if args.debug else "-beta_nov16 -detect_links false -cyclic_peptide -mute all")
    objs = rosetta_scripts.XmlObjects.create_from_file(args.xml)
    fr = objs.get_mover("full_relax_complex")
    pcm = objs.get_mover("pcm")

    jobs = [l.rstrip("\n").split("\t") for l in Path(args.jobs_tsv).read_text().splitlines() if l.strip()]
    n_fail = 0
    for k, (in_pdb, fasta, out_pdb) in enumerate(jobs, 1):
        print(f"[{k}/{len(jobs)}] {in_pdb}", flush=True)
        try:
            run_job(in_pdb, fasta, out_pdb, pcm, fr)
        except Exception:
            n_fail += 1
            print(f"[WARN] job failed for {in_pdb}:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
    print(f"Done: {len(jobs) - n_fail}/{len(jobs)} succeeded", flush=True)


if __name__ == "__main__":
    main()