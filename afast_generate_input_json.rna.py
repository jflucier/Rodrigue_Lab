import argparse
import csv
import json
import os
import string
import sys


def get_extended_chain_ids(start_idx, count):
    """Generates an array of unique chain IDs, supporting multi-character IDs (A-Z, AA-ZZ)."""
    alphabet = string.ascii_uppercase
    chain_ids = []

    for i in range(start_idx, start_idx + count):
        if i < 26:
            chain_ids.append(alphabet[i])
        else:
            first_letter = alphabet[(i // 26) - 1]
            second_letter = alphabet[i % 26]
            chain_ids.append(f"{first_letter}{second_letter}")

    return chain_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AlphaFold3 JSON files from an RNA structural fold list TSV.")
    parser.add_argument("-i", "--input_tsv", required=True, help="Path to the input TSV file specifying the RNA folds.")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Directory path where the generated JSON files will be stored.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input_tsv):
        print(f"Error: Input TSV file not found at '{args.input_tsv}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Dictionary mapping column headers to their corresponding official CCD codes
    ligand_ccd_map = {
        "TPP": "TPP",
        "Mg2+": "MG",
        "Ca2+": "CA",
        "Mn2+": "MN",
        "Na+": "NA",
        "K+": "K"
    }

    with open(args.input_tsv, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        generated_count = 0
        for row_idx, row in enumerate(reader, start=1):
            try:
                sequences_array = []
                current_chain_offset = 0

                # 1. Parse RNA Sequence
                rna_name = (row.get('name') or f"RNA_{row_idx}").strip()
                rna_seq = (row.get('sequence') or "").strip()

                if not rna_seq:
                    print(f"Warning: Skipping row {row_idx} ({rna_name}) because sequence is missing.", file=sys.stderr)
                    continue

                # Generate chain ID for the RNA molecule
                rna_ids = get_extended_chain_ids(current_chain_offset, 1)
                current_chain_offset += 1

                sequences_array.append({
                    "rna": {
                        "id": rna_ids[0],
                        "sequence": rna_seq
                    }
                })

                # 2. Parse Ligands and Ions
                for col_name, ccd_code in ligand_ccd_map.items():
                    count_str = (row.get(col_name) or "0").strip()
                    try:
                        count = int(count_str)
                    except ValueError:
                        count = 0

                    if count > 0:
                        ligand_ids = get_extended_chain_ids(current_chain_offset, count)
                        current_chain_offset += count

                        # Loop to expand instances per unique chain ID block
                        for lig_id in ligand_ids:
                            sequences_array.append({
                                "ligand": {
                                    "id": lig_id,
                                    "ccdCodes": [ccd_code]
                                }
                            })

                # 3. Create JSON payload file
                output_filename = f"{rna_name}.json"
                json_data = {
                    "name": rna_name,
                    "sequences": sequences_array,
                    "modelSeeds": list(range(1, 4)),
                    "dialect": "alphafold3",
                    "version": 3
                }

                full_output_path = os.path.join(args.output_dir, output_filename)
                with open(full_output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(json_data, out_f, indent=2)

                generated_count += 1

            except Exception as e:
                print(f"Warning: Skipping row {row_idx} due to unexpected processing error: {e}", file=sys.stderr)

    print(
        f"Successfully processed sheet. Generated {generated_count} RNA-ligand JSON structures inside: {args.output_dir}")


if __name__ == "__main__":
    main()
