import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Inject complete alternating Switchcraft REMARK 999 headers.")
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

    clean_lines = [l for l in lines if not l.startswith("REMARK 999") and not l.startswith("REMARK    motif_spec")]

    # Core architecture sizing headers
    header_lines = [
        f"REMARK 999 NAME   {motif_name.upper():<4}\n",
        f"REMARK 999 MINIMUM TOTAL LENGTH      200\n",
        f"REMARK 999 MAXIMUM TOTAL LENGTH      220\n"
    ]

    target_chains = ["A", "B"]

    for chain in target_chains:
        # We explicitly interleave designable 'scaffold' regions (no chain letter)
        # and frozen 'motif' regions (with chain letter) so the merge code validates End-to-End.
        architecture = [
            {"type": "scaffold", "min": 5, "max": 50},  # Pad 1: N-Terminus
            {"type": "motif", "min": 49, "max": 64},  # Cluster 1: cAMP face
            {"type": "scaffold", "min": 2, "max": 15},  # Pad 2: Internal loop
            {"type": "motif", "min": 71, "max": 86},  # Cluster 2: cAMP floor
            {"type": "scaffold", "min": 10, "max": 45},  # Pad 3: Internal loop
            {"type": "motif", "min": 123, "max": 136},  # Cluster 3: Hinge link
            {"type": "scaffold", "min": 10, "max": 40},  # Pad 4: Internal loop
            {"type": "motif", "min": 169, "max": 191},  # Cluster 4: DNA Horizon
            {"type": "scaffold", "min": 5, "max": 30}  # Pad 5: C-Terminus (Ensures a trailing scaffold segment!)
        ]

        for segment in architecture:
            if segment["type"] == "scaffold":
                # Space character at index 18 forces Switchcraft to see a 'scaffold'
                header_lines.append(f"REMARK 999 INPUT   {segment['min']:>4}{segment['max']:>4}\n")
            else:
                # Chain character at index 18 forces Switchcraft to see a frozen 'motif'
                header_lines.append(f"REMARK 999 INPUT  {chain}{segment['min']:>4}{segment['max']:>4}\n")

    final_header = "".join(header_lines)

    with open(args.output, 'w') as f:
        f.write(final_header)
        f.writelines(clean_lines)

    print(f"Success! Properly padded mergeable template saved to: {args.output}")


if __name__ == "__main__":
    main()