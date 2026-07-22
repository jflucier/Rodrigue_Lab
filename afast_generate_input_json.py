import argparse
import csv
import json
import os
import string
import sys


def get_chain_ids(start_idx, count):
    """Generates an array of alphabet chain IDs based on a starting position and count."""
    alphabet = string.ascii_uppercase
    if start_idx + count > len(alphabet):
        print(
            f"Error: Requested {count} chains starting from index {start_idx}, which exceeds available alphabet letters.",
            file=sys.stderr)
        sys.exit(1)
    return [alphabet[i] for i in range(start_idx, start_idx + count)]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AlphaFold3 JSON files from a structural fold list TSV.")
    parser.add_argument("-i", "--input_tsv", required=True, help="Path to the input TSV file specifying the folds.")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Directory path where the generated JSON files will be stored.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify input file exists
    if not os.path.isfile(args.input_tsv):
        print(f"Error: Input TSV file not found at '{args.input_tsv}'", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory structure exists
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_tsv, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        # Verify required headers are present in the TSV file
        required_headers = [
            'protein1_name', 'protein1_nbr', 'protein1_seq',
            'protein2_name', 'protein2_nbr', 'protein2_seq'
        ]
        missing = [h for h in required_headers if h not in reader.fieldnames]
        if missing:
            print(f"Error: Input TSV is missing required column headers: {missing}", file=sys.stderr)
            sys.exit(1)

        generated_count = 0
        for row_idx, row in enumerate(reader, start=1):
            try:
                p1_name = row['protein1_name']
                p2_name = row['protein2_name']
                p1_seq = row['protein1_seq'].strip()
                p2_seq = row['protein2_seq'].strip()

                # Convert numeric strings to counts
                p1_count = int(row['protein1_nbr'])
                p2_count = int(row['protein2_nbr'])

                # Dynamically calculate non-overlapping chain letters (e.g., A,B and then C)
                p1_ids = get_chain_ids(0, p1_count)
                p2_ids = get_chain_ids(p1_count, p2_count)

                # Construct JSON schema matching AlphaFold3 specification
                json_data = {
                    "name": f"{p1_name}__{p2_name}",
                    "sequences": [
                        {
                            "protein": {
                                "id": p1_ids,
                                "sequence": p1_seq
                            }
                        },
                        {
                            "protein": {
                                "id": p2_ids,
                                "sequence": p2_seq
                            }
                        }
                    ],
                    "modelSeeds": [1, 2, 3],
                    "dialect": "alphafold3",
                    "version": 3
                }

                # Create isolated destination filepath
                output_filename = f"{p1_name}__{p2_name}.json"
                full_output_path = os.path.join(args.output_dir, output_filename)

                with open(full_output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(json_data, out_f, indent=2)

                generated_count += 1

            except ValueError:
                print(f"Warning: Skipping row {row_idx} due to non-integer values in protein_nbr columns.",
                      file=sys.stderr)
            except Exception as e:
                print(f"Warning: Skipping row {row_idx} due to unexpected processing error: {e}", file=sys.stderr)

        print(
            f"Successfully processed matrix list. Generated {generated_count} JSON structural payloads inside: {args.output_dir}")


if __name__ == "__main__":
    main()
