import json
import os
import argparse


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Inject Switchcraft motif_spec headers into a PDB file safely.")
    parser.add_argument("-i", "--input", required=True, help="Path to the original source PDB file")
    parser.add_argument("-o", "--output", required=True, help="Path where the new annotated PDB file should be saved")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Extract a clean motif name from the output filename
    base_name = os.path.basename(args.output)
    motif_name = os.path.splitext(base_name)[0].lower()

    # Read from the raw untouched source file
    with open(args.input, 'r') as f:
        lines = f.readlines()

    # Clean out any old tracking headers if present
    clean_lines = [l for l in lines if not l.startswith("REMARK    ") or "motif_spec" not in l]

    # Map the four precise contact clusters for Chain A and Chain B
    target_chains = ["A", "B"]
    segments = []

    for chain in target_chains:
        segments.extend([
            {"chain": chain, "start": 49, "end": 64},  # Cluster 1: cAMP pocket face
            {"chain": chain, "start": 71, "end": 86},  # Cluster 2: cAMP pocket floor
            {"chain": chain, "start": 123, "end": 136},  # Cluster 3: cAMP hinge link
            {"chain": chain, "start": 169, "end": 191}  # Cluster 4 + DNA Horizon (Merged)
        ])

    spec = {
        "name": motif_name,
        "segments": segments
    }

    header = f"REMARK    motif_spec {json.dumps(spec)}\n"

    # Write everything to the new specified output path
    with open(args.output, 'w') as f:
        f.write(header)
        f.writelines(clean_lines)

    print(f"Success! Annotated motif saved to: {args.output}")


if __name__ == "__main__":
    main()