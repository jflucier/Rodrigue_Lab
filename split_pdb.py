import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Extract, relabel, and normalize PDB chains into separate files, with automatic chain detection if unspesified."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input complex PDB file"
    )
    parser.add_argument(
        "-o", "--outdir", required=True, help="Directory path where individual PDB files should be saved"
    )
    parser.add_argument(
        "-c", "--chains", default=None,
        help="Optional: Comma-separated list of chain characters to extract. If omitted, all chains are extracted."
    )
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.outdir

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    with open(input_file, "r") as f:
        lines = f.readlines()

    # --- Phase 1: Mirroring and Normalization Pass ---
    # To ensure mirror-relabeling logic is applied correctly to the entire dataset,
    # we process and normalize the lines in a first pass while tracking seen atoms.
    processed_lines = []
    detected_chains = set()
    seen_atoms = set()

    for line in lines:
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            atom_id = line[6:11].strip()
            chain = line[21:22]

            # Build a unique tracking key for Atom ID + Original Chain Letter
            key = (atom_id, chain)

            # Mirror copy relabeling correction
            if key in seen_atoms:
                if chain == "B":
                    line = line[:21] + "C" + line[22:]
                elif chain == "E":
                    line = line[:21] + "F" + line[22:]
                elif chain == "A":
                    line = line[:21] + "D" + line[22:]
            else:
                seen_atoms.add(key)

            final_chain = line[21:22].strip()
            # If the chain is empty/whitespace, track it as an unlabelled entry placeholder
            if final_chain == "":
                final_chain = "_"
            detected_chains.add(final_chain)

            is_hetatm = line.startswith("HETATM")
            # CONVERSION LOGIC: Transform non-standard HETATM amino acids into standard ATOM rows
            if is_hetatm:
                line = "ATOM  " + line[6:]
                if "MSE" in line:
                    line = line[:17] + "MET" + line[20:]
                    if "SE " in line:
                        line = line.replace("SE ", "SD ")
                elif "SEC" in line:
                    line = line[:17] + "CYS" + line[20:]
                    if "SE " in line:
                        line = line.replace("SE ", "SG ")

            processed_lines.append((final_chain, line))

    # --- Phase 2: Establish Target Extraction Criteria ---
    if args.chains:
        # User provided explicit targets via command line
        target_chains = [c.strip().upper() for c in args.chains.split(",") if c.strip()]
        print(f"Extracting user-specified target chains: {target_chains}")
    else:
        # Generic fallback behavior: extract every unique entity detected in the file
        target_chains = sorted(list(detected_chains))
        print(f"No specific chains provided. Auto-detected and extracting ALL chains: {target_chains}")

    # Allocate empty storage arrays for the targets
    chain_bins = {chain_id: [] for chain_id in target_chains}

    # Populate the storage arrays using our normalized coordinate records
    for final_chain, line in processed_lines:
        if final_chain in chain_bins:
            chain_bins[final_chain].append(line)

    # Automatically ensure output destination path exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Phase 3: File Compilation Output Pass ---
    for chain_id in target_chains:
        clean_lines = chain_bins[chain_id]
        if len(clean_lines) == 0:
            print(f"Warning: No atom records isolated for chain '{chain_id}'")
            continue

        file_label = chain_id.lower() if chain_id != "_" else "unlabelled"
        file_name = f"chain_{file_label}.pdb"
        target_path = os.path.join(output_dir, file_name)

        with open(target_path, "w") as out:
            out.writelines(clean_lines)
            out.write("TER\n")
            out.write("END\n")

        print(f" -> Saved isolated chain component ({len(clean_lines)} lines): {target_path}")

    print(f"\nSuccess! Fragment files generated cleanly inside: {output_dir}/")


if __name__ == "__main__":
    main()
