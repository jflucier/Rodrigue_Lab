import os
import argparse


def extract_rpoa_chains(pdb_path, target_chains):
    rpoa_lines = []
    if not os.path.exists(pdb_path):
        print(f"Error: Reference source file '{pdb_path}' not found.")
        return rpoa_lines

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                chain_id = line[21:22].strip()
                if chain_id in target_chains:
                    rpoa_lines.append(line)
    return rpoa_lines


def main():
    parser = argparse.ArgumentParser(
        description="Stitch isolated design backbones with complete multi-chain biological assemblies for LigandMPNN sequence conditioning."
    )
    parser.add_argument("-d", "--design", required=True,
                        help="Path to your Switchcraft generated state4_sampleX.pdb file")
    parser.add_argument("-r", "--reference", required=True,
                        help="Path to your 1lb2.complex.relabelled.pdb reference file containing all 4 RpoA domains")
    parser.add_argument("-o", "--output", required=True,
                        help="Path where the final complete composite PDB file should be saved")
    parser.add_argument("-c", "--chains", default="B,C,E,F",
                        help="Comma-separated list of target RpoA chain letters to extract from reference (Default: B,C,E,F)")

    args = parser.parse_args()
    target_chains = [c.strip().upper() for c in args.chains.split(",")]

    if not os.path.exists(args.design):
        print(f"Error: Design file '{args.design}' not found.")
        return

    # Read design layers
    with open(args.design, "r") as f:
        design_lines = f.readlines()

    # Isolate strictly your designed core protein chains (A and B), DNA, and cAMP (CMP)
    # This automatically discards the single pruned alpha helix Switchcraft leftover fragment
    clean_design_lines = []
    for line in design_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            chain_id = line[21:22].strip()
            # Retain engineered CRP chains or non-protein groups (ligands/nucleic acids)
            if chain_id in ["A", "B"] or chain_id == "":
                clean_design_lines.append(line)
        elif not line.startswith("END"):
            clean_design_lines.append(line)

    print(f"Extracting targets {target_chains} from biological assembly reference...")
    rpoa_lines = extract_rpoa_chains(args.reference, target_chains)

    if len(rpoa_lines) == 0:
        print("Error: Extracted 0 RpoA lines. Verify your reference file path and chain letters.")
        return

    print(f"Extracted {len(rpoa_lines)} native partner coordinate entries.")

    # Build directory path if it does not exist
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write unified complete structural landscape
    with open(args.output, "w") as out:
        out.writelines(clean_design_lines)
        if not clean_design_lines[-1].endswith("\n"):
            out.write("\n")
        out.write("TER\n")
        out.writelines(rpoa_lines)
        out.write("END\n")

    print(f"Success! Complete structural model generated safely at: {args.output}")


if __name__ == "__main__":
    main()
