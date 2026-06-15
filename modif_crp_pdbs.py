import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Inject valid Switchcraft headers and normalize HETATMs while strictly calculating target lengths from protein chains."
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

    clean_lines = [l for l in lines if not l.startswith("REMARK 999") and not l.startswith("REMARK    motif_spec")]

    target_chains = ["A", "B"]
    motif_atom_lines = []
    unique_residues = set()

    for line in clean_lines:
        is_atom = line.startswith("ATOM  ")
        is_hetatm = line.startswith("HETATM")

        if is_atom or is_hetatm:
            if len(line) < 27:
                continue

            # Standard PDB character coordinates slicing
            chain_id = line[21:22].strip()

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

    # DYNAMIC LOGIC: Calculate exact matching dimensions from true disk presence
    total_residues = len(unique_residues)
    if total_residues == 0:
        print("Error: No target residues found in the specified chains.")
        return

    # Find the true starting and ending points for each chain to prevent clipping
    chain_a_res = [r for c, r in unique_residues if c == 'A']
    chain_b_res = [r for c, r in unique_residues if c == 'B']

    a_min, a_max = min(chain_a_res), max(chain_a_res)
    b_min, b_max = min(chain_b_res), max(chain_b_res)

    # Set boundaries with structural padding buffers
    min_len = total_residues - 4
    max_len = total_residues + 46

    print(f"Dynamically mapped {total_residues} total protein residues.")
    print(f"Chain A: {a_min} to {a_max} ({len(chain_a_res)} residues)")
    print(f"Chain B: {b_min} to {b_max} ({len(chain_b_res)} residues)")

    header_lines = [
        f"REMARK 999 NAME   {motif_name.upper():<4}\n",
        f"REMARK 999 PDB    {motif_name.upper():<4}\n",
        f"REMARK 999 INPUT  A  {a_min:>2} {a_max:>3}\n",
        f"REMARK 999 INPUT  B  {b_min:>2} {b_max:>3}\n",
        f"REMARK 999 MINIMUM TOTAL LENGTH      {min_len}\n",
        f"REMARK 999 MAXIMUM TOTAL LENGTH      {max_len}\n"
    ]

    with open(args.output, 'w') as f:
        f.write("".join(header_lines))
        f.writelines(motif_atom_lines)

    print(f"Success! Perfect mathematical background template saved to: {args.output}")


if __name__ == "__main__":
    main()
