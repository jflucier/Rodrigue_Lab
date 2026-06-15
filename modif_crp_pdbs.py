import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Inject valid Switchcraft length-padded REMARK 999 headers and precisely filter atoms using strict PDB columns including rpoA interface."
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

    # Architecture layout definition updated to isolate rpoA contact motif
    architecture = [
        {"type": "scaffold", "min": 10, "max": 50},  # Pad 1 length: handles residues 1-48
        {"type": "motif", "min": 49, "max": 64},  # Cluster 1 indices: cAMP face
        {"type": "scaffold", "min": 2, "max": 10},  # Pad 2 length: handles residues 65-70
        {"type": "motif", "min": 71, "max": 86},  # Cluster 2 indices: cAMP floor
        {"type": "scaffold", "min": 10, "max": 40},  # Pad 3 length: handles residues 87-122
        {"type": "motif", "min": 123, "max": 136},  # Cluster 3 indices: Hinge link

        # === SPLIT ORIGINAL PAD 4 (137-168) TO INTEGRATE RPOA DOCK SURFACE ===
        {"type": "scaffold", "min": 5, "max": 20},  # Pad 4a length: handles residues 137-155
        {"type": "motif", "min": 156, "max": 164},  # Cluster 4 indices: rpoA contact motif
        {"type": "scaffold", "min": 2, "max": 10},  # Pad 4b length: handles residues 165-168

        {"type": "motif", "min": 169, "max": 191},  # Cluster 5 indices: DNA Helix
        {"type": "scaffold", "min": 5, "max": 25}  # Pad 5 length: handles residues 192-210
    ]

    # Generate REMARK 999 strings
    for chain in target_chains:
        for segment in architecture:
            if segment["type"] == "scaffold":
                header_lines.append(f"REMARK 999 INPUT   {segment['min']:>4}{segment['max']:>4}\n")
            else:
                header_lines.append(f"REMARK 999 INPUT  {chain}{segment['min']:>4}{segment['max']:>4}\n")

    final_header = "".join(header_lines)

    # Filter structural lines using standard PDB coordinate fixed-width rules
    motif_atom_lines = []
    motif_ranges = [seg for seg in architecture if seg["type"] == "motif"]

    for line in clean_lines:
        is_atom = line.startswith("ATOM  ")
        is_hetatm = line.startswith("HETATM")

        if is_atom or is_hetatm:
            if len(line) < 27:
                continue

            # Standard PDB Rules: Chain ID is index 21, Residue Seq is indices 22-26
            chain_id = line[21:22].strip()
            try:
                res_seq = int(line[22:26].strip())
            except ValueError:
                continue

            # Keep atom only if it belongs to Chain A or B AND falls inside a motif window
            if chain_id in target_chains:
                is_inside_motif = any(
                    seg["min"] <= res_seq <= seg["max"]
                    for seg in motif_ranges
                )
                if is_inside_motif:
                    # Normalize non-standard HETATM amino acids into standard ATOM entries on-the-fly
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

    # Write out the structural file containing headers and properly pruned atoms
    with open(args.output, 'w') as f:
        f.write(final_header)
        f.writelines(motif_atom_lines)

    print(f"Success! Normalized template with rpoA interface saved to: {args.output}")


if __name__ == "__main__":
    main()
