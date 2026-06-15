import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Inject valid Switchcraft headers and normalize HETATMs while dynamically calculating length parameters from coordinates."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the original source PDB file")
    parser.add_argument("-o", "--output", required=True, help="Path where the new annotated PDB file should be saved")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    base_name = os.path.basename(args.output)
    motif_name = os.path.splitext(base_name)[0].lower()

    with open(args.input, 'r') as f:
        lines = f.readlines()

    # Clean existing headers
    clean_lines = [l for l in lines if not l.startswith("REMARK 999") and not l.startswith("REMARK    motif_spec")]

    target_chains = ["A", "B"]
    motif_atom_lines = []

    # Track unique tracking tuples (chain_id, res_seq) to calculate exact size
    unique_residues = set()

    # Process and clean lines first to get accurate coordinates
    for line in clean_lines:
        is_atom = line.startswith("ATOM  ")
        is_hetatm = line.startswith("HETATM")

        if is_atom or is_hetatm:
            if len(line) < 27:
                continue

            # Standard PDB character coordinates slicing
            chain_id = line[21:22].strip()

            # Keep ALL residues belonging to your target protein chains
            if chain_id in target_chains:
                try:
                    res_seq = int(line[22:26].strip())
                    unique_residues.add((chain_id, res_seq))
                except ValueError:
                    pass

                if is_hetatm:
                    # Convert line type identifier
                    line = "ATOM  " + line[6:]

                    # Normalize Selenomethionine (MSE -> MET)
                    if "MSE" in line:
                        line = line[:17] + "MET" + line[20:]
                        if "SE " in line:
                            line = line.replace("SE ", "SD ")

                    # Normalize Selenocysteine (SEC -> CYS)
                    elif "SEC" in line:
                        line = line[:17] + "CYS" + line[20:]
                        if "SE " in line:
                            line = line.replace("SE ", "SG ")

                motif_atom_lines.append(line)

    # 🔥 DYNAMIC CALCULATION SECTION 🔥
    total_residues = len(unique_residues)
    if total_residues == 0:
        print("Error: No target residues found in the specified chains.")
        return

    # Automatically set responsive bounds based on the true physical chain length
    min_len = total_residues - 4
    max_len = total_residues + 46

    print(f"Dynamically mapped {total_residues} residues. Setting target limits to: {min_len} - {max_len}")

    # Core architecture sizing headers matching native git-cloned formats
    header_lines = [
        f"REMARK 999 NAME   {motif_name.upper():<4}\n",
        f"REMARK 999 PDB    {motif_name.upper():<4}\n",
        f"REMARK 999 INPUT  A   9 210\n",
        f"REMARK 999 INPUT  B   9 210\n",
        f"REMARK 999 MINIMUM TOTAL LENGTH      {min_len}\n",
        f"REMARK 999 MAXIMUM TOTAL LENGTH      {max_len}\n"
    ]

    # Write out the complete file containing all background sequences
    with open(args.output, 'w') as f:
        f.write("".join(header_lines))
        f.writelines(motif_atom_lines)

    print(f"Success! Complete dynamic background template saved to: {args.output}")


if __name__ == "__main__":
    main()
