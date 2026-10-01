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
from io import StringIO

from pyrosetta import init, pose_from_pdb
from pyrosetta.io import pose_from_pdbstring
from pyrosetta.rosetta.core.scoring import CA_rmsd
import pyrosetta.rosetta.protocols.rosetta_scripts as rosetta_scripts
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta.rosetta.protocols.simple_moves import CyclizationMover

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

def load_pose_with_ter_fix(pdb_path):
    """
    Parses a raw PDB file and injects a TER record between Chain A and Chain B
    to prevent Rosetta from bridging polymer bonds over open target protein gaps.
    """
    raw_lines = Path(pdb_path).read_text().splitlines()
    fixed_lines = []
    last_chain = None

    for line in raw_lines:
        if line.startswith("ATOM  "):
            current_chain = line[21]  # Extract Chain ID column
            if last_chain == "A" and current_chain == "B":
                fixed_lines.append("TER")
            last_chain = current_chain
        fixed_lines.append(line)

    return pose_from_pdbstring("\n".join(fixed_lines))

def get_previous_round_path(out_pdb):
    """
    Deduces the path of the previous round's output file to compare RMSD.
    Example: if out_pdb ends with '..._mpnn4.pdb', looks for '..._mpnn3.pdb'
    """
    out_path = Path(out_pdb)
    name = out_path.stem

    # Check if the name ends with an explicit round tracker index suffix
    for i in range(2, 6):  # Checks rounds 2, 3, 4, 5
        if name.endswith(f"_mpnn{i}"):
            prev_name = name.replace(f"_mpnn{i}", f"_mpnn{i - 1}")
            prev_path = out_path.with_name(prev_name + out_path.suffix)
            if prev_path.exists():
                return prev_path
    return None

def run_job(in_pdb, fasta, out_pdb, fr):
    seq = read_designed_sequence(fasta)

    raw_lines = Path(in_pdb).read_text().splitlines()
    fixed_lines = []
    last_chain = None

    for line in raw_lines:
        if line.startswith("ATOM  "):
            current_chain = line[21]  # Extract the Chain ID column
            # If we just finished reading Chain A and are moving to Chain B, insert a TER record
            if last_chain == "A" and current_chain == "B":
                fixed_lines.append("TER")
            last_chain = current_chain
        fixed_lines.append(line)

    # pose = pose_from_pdb(in_pdb)
    pdb_string_data = "\n".join(fixed_lines)
    print(f"PDB: {pdb_string_data}")
    pose = pose_from_pdbstring(pdb_string_data)

    info = pose.pdb_info()
    chain_a = [i for i in range(1, pose.size() + 1) if info.chain(i) == "A"]

    if len(seq) != len(chain_a):
        raise ValueError(f"designed seq length {len(seq)} != chain A length "
                         f"{len(chain_a)} in {in_pdb}")
    # bad = sorted(set(seq) - set(AA_1TO3))
    # if bad:
    #     raise ValueError(f"non-standard characters in designed sequence: {bad}")

    # gap = terminus_gap(pose, chain_a)
    # print(f"  threading {seq} onto chain A ({len(chain_a)} res), "
    #       f"N-C terminus gap = {gap:.2f} A (expect ~1.3 A for a closed bond)",
    #       flush=True)
    # if gap > 3.0:
    #     print(f"  [NOTE] gap is large -- this backbone's termini likely never "
    #           f"closed during RFdiffusion sampling; a cyclization failure below "
    #           f"is expected for this one, not a pipeline bug", flush=True)

    mutator = MutateResidue()
    for pose_idx, aa in zip(chain_a, seq):
        mutator.set_target(pose_idx)
        mutator.set_res_name(AA_1TO3[aa])
        mutator.apply(pose)

    # 2. OVERRIDE THE XML AUTO-DETECTION: Explicitly lock the 13-residue cycle
    # Syntax: CyclizationMover( chain_number, add_constraints, minimize, build_conformation )
    python_cyclizer = CyclizationMover(1, True, True, 1)
    python_cyclizer.apply(pose)

    prev_pdb_path = get_previous_round_path(out_pdb)
    if prev_pdb_path:
        try:
            prev_pose = load_pose_with_ter_fix(prev_pdb_path)
            drift = CA_rmsd(pose, prev_pose, 1, len(chain_a))
            print(f"  --> Macrocycle backbone drift vs. previous round: {drift:.4f} A", flush=True)
        except Exception:
            print("  --> [NOTE] Failed to calculate backbone RMSD comparison.", flush=True)

    # pcm.apply(pose)
    # 3. Execute your interface relaxation safely
    fr.apply(pose)
    # pcm.apply(pose)
    # 4. Re-verify the loop constraints at the true 13-residue index endpoint
    python_cyclizer.apply(pose)

    sio = StringIO()
    pose.dump_pdb(sio)
    rosetta_lines = sio.getvalue().splitlines()

    # 2. Filter out non-standard structural records (like pose energy tables)
    clean_pdb_lines = [
        line for line in rosetta_lines
        if line.startswith(("ATOM", "HETATM", "TER", "ENDMDL", "END"))
    ]

    # 3. Save a clean, standard PDB file that downstream tools can parse perfectly
    Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
    Path(out_pdb).write_text("\n".join(clean_pdb_lines))

    # Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
    # pose.dump_pdb(out_pdb)


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

    init("-beta_nov16" if args.debug else "-beta_nov16 -mute all")
    objs = rosetta_scripts.XmlObjects.create_from_file(args.xml)
    fr = objs.get_mover("full_relax_complex")
    # pcm = objs.get_mover("pcm")

    jobs = [l.rstrip("\n").split("\t") for l in Path(args.jobs_tsv).read_text().splitlines() if l.strip()]
    n_fail = 0
    for k, (in_pdb, fasta, out_pdb) in enumerate(jobs, 1):
        print(f"[{k}/{len(jobs)}] {in_pdb}", flush=True)
        try:
            # run_job(in_pdb, fasta, out_pdb, pcm, fr)
            run_job(in_pdb, fasta, out_pdb, fr)
        except Exception:
            n_fail += 1
            print(f"[WARN] job failed for {in_pdb}:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
    print(f"Done: {len(jobs) - n_fail}/{len(jobs)} succeeded", flush=True)


if __name__ == "__main__":
    main()