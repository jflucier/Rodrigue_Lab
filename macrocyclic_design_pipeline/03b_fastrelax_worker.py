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

--isolate runs every job in its own fresh Python/PyRosetta process. Use it if a
design fails inside a chunk but succeeds when run alone (PyRosetta keeps global
residue-type state across jobs), or to survive Rosetta calls that kill the
whole process. It costs one PyRosetta init + XML parse per job, small next to
FastRelax.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from io import StringIO

from pyrosetta import init, pose_from_pdb
from pyrosetta.io import pose_from_pdbstring
from pyrosetta.rosetta.core.scoring import CA_rmsd

import pyrosetta.rosetta.core.pose as core_pose
import pyrosetta.rosetta.core.import_pose as import_pose
import pyrosetta.rosetta.core.chemical as core_chemical
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

def describe_chain_a(pose, chain_a, seq):
    """Diagnostics printed when the first cyclization step fails (best effort)."""
    try:
        print(f"  [DIAG] chain A: {len(chain_a)} residues, designed seq {seq}", flush=True)
        for label, idx in (("first", chain_a[0]), ("last", chain_a[-1])):
            r = pose.residue(idx)
            print(f"  [DIAG] {label}: pose {idx} {r.type().name()} "
                  f"lower_terminus={r.is_lower_terminus()} upper_terminus={r.is_upper_terminus()}",
                  flush=True)
        print(f"  [DIAG] N(first)-C(last) distance = {terminus_gap(pose, chain_a):.2f} A", flush=True)
        gaps = [(pose.residue(a).xyz("C") - pose.residue(b).xyz("N")).norm()
                for a, b in zip(chain_a[:-1], chain_a[1:])]
        print(f"  [DIAG] max consecutive C(i)-N(i+1) distance = {max(gaps):.2f} A "
              f"(~1.33 expected)", flush=True)
    except Exception as e:
        print(f"  [DIAG] could not collect diagnostics: {e}", flush=True)


def load_pose_with_ter_fix(pdb_path):
    """
    Parses a raw PDB file and injects a TER record between Chain A and Chain B
    to prevent Rosetta from bridging polymer bonds over open target protein gaps.
    """
    raw_lines = Path(pdb_path).read_text().splitlines()
    fixed_lines = []
    last_chain = None

    for line in raw_lines:
        # Prevent any leftover connection definitions from bleeding into the new map
        if line.startswith(("LINK", "CONECT", "SSBOND")):
            continue

        if line.startswith("ATOM  "):
            current_chain = line[21]  # Extract Chain ID column
            if last_chain == "A" and current_chain == "B":
                fixed_lines.append("TER")
            last_chain = current_chain
        fixed_lines.append(line)

    pdb_string_data = "\n".join(fixed_lines)
    # 1. Create a completely empty, uninitialized pose container
    pose = core_pose.Pose()

    # 2. Query a fresh, untouched copy of the standard all-atom chemical properties
    # This bypasses the corrupted residue variations lingering in memory from previous iterations
    clean_type_set = core_chemical.ChemicalManager.get_instance().residue_type_set("fa_standard")

    # 3. Read the coordinates straight into the pose using our isolated chemical slate
    import_pose.pose_from_pdbstring(pose, pdb_string_data, clean_type_set, "string_stream")

    return pose

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

    # Safely creates an uncorrupted structural layout via our fresh type-set injector
    pose = load_pose_with_ter_fix(in_pdb)

    info = pose.pdb_info()
    chain_a = [i for i in range(1, pose.size() + 1) if info.chain(i) == "A"]

    if len(seq) != len(chain_a):
        raise ValueError(f"designed seq length {len(seq)} != chain A length "
                         f"{len(chain_a)} in {in_pdb}")

    mutator = MutateResidue()
    for pose_idx, aa in zip(chain_a, seq):
        mutator.set_target(pose_idx)
        mutator.set_res_name(AA_1TO3[aa])
        mutator.apply(pose)

    # # 2. OVERRIDE THE XML AUTO-DETECTION: Explicitly lock the 13-residue cycle
    # # Syntax: CyclizationMover( chain_number, add_constraints, minimize, build_conformation )
    # python_cyclizer = CyclizationMover(1, True, True, 1)
    # try:
    #     print("bef")
    #     python_cyclizer.apply(pose)
    #     print("after")
    # except Exception:
    #     describe_chain_a(pose, chain_a, seq)
    #     raise

    import pyrosetta.rosetta.core.conformation as core_conformation
    # 2. Bypassing CyclizationMover: Explicitly link head-to-tail atoms
    start_res = chain_a[0]  # Typically residue 1
    end_res = chain_a[-1]  # Your true macrocycle endpoint (e.g., residue 11)

    # Establish the physical covalent bond between terminal Nitrogen and Carbon atoms
    pose.conformation().declare_chemical_bond(start_res, "N", end_res, "C")

    # 3. Add the variant types using your confirmed signature layout
    # Matches: (conformation, variant_type, seqpos)
    core_conformation.add_variant_type_to_conformation_residue(
        pose.conformation(),
        core_chemical.VariantType.CUTPOINT_UPPER,
        start_res
    )
    core_conformation.add_variant_type_to_conformation_residue(
        pose.conformation(),
        core_chemical.VariantType.CUTPOINT_LOWER,
        end_res
    )

    # 4. Synchronize non-ideal virtual cutpoint atoms across the covalent loop junction
    # This recalculates C-OVL1-OVL2 and OVU1-N distances so the score function is stable
    core_conformation.update_cutpoint_virtual_atoms_if_connected(pose.conformation(), start_res, True)
    core_conformation.update_cutpoint_virtual_atoms_if_connected(pose.conformation(), end_res, True)

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

    pose.dump_pdb(out_pdb)


def run_isolated(jobs, args):
    """One fresh interpreter per job. Returns the number of failed jobs."""
    script = str(Path(__file__).resolve())
    n_fail = 0
    for k, job in enumerate(jobs, 1):
        print(f"[{k}/{len(jobs)}] {job[0]}  (isolated process)", flush=True)
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
            fh.write("\t".join(job) + "\n")
            tmp = fh.name
        cmd = [sys.executable, script, "--jobs-tsv", tmp, "--xml", args.xml, "--child"]
        if args.debug:
            cmd.append("--debug")
        try:
            rc = subprocess.run(cmd).returncode
        finally:
            os.unlink(tmp)
        if rc != 0:
            n_fail += 1
            print(f"[WARN] isolated job failed (exit code {rc}) for {job[0]}", flush=True)
    return n_fail


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
    ap.add_argument("--isolate", action="store_true",
                    help="Run each job in a fresh Python/PyRosetta process (see module docstring)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    jobs = [l.rstrip("\n").split("\t") for l in Path(args.jobs_tsv).read_text().splitlines() if l.strip()]

    if args.isolate and not args.child:
        n_fail = run_isolated(jobs, args)
        print(f"Done: {len(jobs) - n_fail}/{len(jobs)} succeeded", flush=True)
        return

    init("-beta_nov16" if args.debug else "-beta_nov16 -mute all")
    objs = rosetta_scripts.XmlObjects.create_from_file(args.xml)
    fr = objs.get_mover("full_relax_complex")
    # pcm = objs.get_mover("pcm")

    n_fail = 0
    for k, (in_pdb, fasta, out_pdb) in enumerate(jobs, 1):
        if not args.child:
            print(f"[{k}/{len(jobs)}] {in_pdb}", flush=True)
        try:
            # run_job(in_pdb, fasta, out_pdb, pcm, fr)
            run_job(in_pdb, fasta, out_pdb, fr)
        except Exception:
            n_fail += 1
            print(f"[WARN] job failed for {in_pdb}:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
    if args.child:
        sys.exit(1 if n_fail else 0)
    print(f"Done: {len(jobs) - n_fail}/{len(jobs)} succeeded", flush=True)


if __name__ == "__main__":
    main()