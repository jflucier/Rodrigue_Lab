import os
import argparse


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Inject Switchcraft REMARK 999 headers into a PDB file safely.")
    parser.add_argument("-i", "--input", required=True, help="Path to the original source PDB file")
    parser.add_argument("-o", "--output", required=True, help="Path where the new annotated PDB file should be saved")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Extract a clean motif name from the output filename (stripping out extensions cleanly)
    base_name = os.path.basename(args.output)
    motif_name = os.path.splitext(base_name)[0].lower()

    # Read from the raw untouched source file
    with open(args.input, 'r') as f:
        lines = f.readlines()

    # Clean out any old tracking headers if present (removes old JSON or 999 remarks)
    clean_lines = [l for l in lines if not l.startswith("REMARK 999") and not l.startswith("REMARK    motif_spec")]

    # Build the required top metadata block exactly matching fixed string requirements
    header_lines = [
        f"REMARK 999 NAME   {motif_name.upper():<4}\n",
        f"REMARK 999 MINIMUM TOTAL LENGTH      200 \n",
        f"REMARK 999 MAXIMUM TOTAL LENGTH      220 \n"
    ]

    # Map the four precise contact clusters for Chain A and Chain B
    target_chains = ["A", "B"]

    for chain in target_chains:
        # Array matching the exact clusters we found in your annotations table
        segments = [
            (49, 64),  # Cluster 1: cAMP pocket face
            (71, 86),  # Cluster 2: cAMP pocket floor
            (123, 136),  # Cluster 3: cAMP hinge link
            (169, 191)  # Cluster 4 + DNA Helix Horizon (Merged)
        ]

        for start, end in segments:
            # Enforces character-perfect spacing for the fixed-column text parser:
            # line[18] is the Chain ID
            # line[19:23] is the right-justified start index
            # line[23:27] is the right-justified end index
            header_lines.append(f"REMARK 999 INPUT  {chain}{start:>4}{end:>4}\n")

    # Combine the new headers with the clean structural data
    final_header = "".join(header_lines)

    # Write everything to the new specified output path without overwriting the source
    with open(args.output, 'w') as f:
        f.write(final_header)
        f.writelines(clean_lines)

    print(f"Success! Correctly formatted REMARK 999 template saved to: {args.output}")


if __name__ == "__main__":
    main()