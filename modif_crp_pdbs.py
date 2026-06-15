import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Inject valid Switchcraft headers and normalize HETATMs while KEEPING full background chains for global motifs."
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

    # Core architecture sizing headers
    header_lines = [
        f"REMARK 999 NAME   {motif_name.upper():<4}\n",
        f"REMARK 999 MINIMUM TOTAL LENGTH      200\n",
        f"REMARK 999 MAXIMUM TOTAL LENGTH      250\n"
    ]

    target_chains = ["A", "B"]

    # Architecture layout flattened to avoid variable-length sampling infinite loops
    # The entire CRP monomer sequence is treated as a continuous, fixed-length motif envelope
    architecture = [
        {"type": "motif", "min": 9, "max": 210}
    ]

    # Generate REMARK 999 strings
    for chain in target_chains:
        for segment in architecture:
            if segment["type"] == "scaffold":
                header_lines.append(f"REMARK 999 INPUT   {segment['min']:>4}{segment['max']:>4}\n")
            else:
                header_lines.append(f"REMARK 999 INPUT  {chain}{segment['min']:>4}{segment['max']:>4}\n")

    final_header = "".join(header_lines)

    motif_atom_lines = []

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

    # Write out the complete file containing all background sequences
    with open(args.output, 'w') as f:
        f.write(final_header)
        f.writelines(motif_atom_lines)

    print(f"Success! Complete background template saved to: {args.output}")


if __name__ == "__main__":
    main()
