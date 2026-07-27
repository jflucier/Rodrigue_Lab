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

        generated_count = 0
        for row_idx, row in enumerate(reader, start=1):
            try:
                sequences_array = []
                current_chain_offset = 0
                last_protein_name = "unknown"

                # Check consecutive numbers dynamically starting from 1 for this specific row
                p_idx = 1
                while True:
                    name_key = f"protein{p_idx}_name"
                    seq_key = f"protein{p_idx}_seq"
                    nbr_key = f"protein{p_idx}_nbr"

                    # Break the inner cell loop as soon as no column keys exist for this number index
                    if name_key not in row:
                        break

                    p_name = (row[name_key] or "").strip()
                    p_seq = (row[seq_key] or "").strip()
                    p_nbr_str = (row[nbr_key] or "").strip()

                    # If this optional slot column exists but is completely blank in this specific row,
                    # check if higher indices exist (safeguard against empty middle column values)
                    if not p_name and not p_seq:
                        p_idx += 1
                        # Dynamic boundary break if we exceed the header keys length
                        if p_idx > len(row):
                            break
                        continue

                    # Fallback assignment for stoichiometric counts
                    p_count = int(p_nbr_str) if p_nbr_str else 1

                    # Map continuous non-overlapping alphabet letters across partners
                    p_ids = get_chain_ids(current_chain_offset, p_count)
                    current_chain_offset += p_count

                    last_protein_name = p_name

                    sequences_array.append({
                        "protein": {
                            "id": p_ids,
                            "sequence": p_seq
                        }
                    })

                    p_idx += 1

                # Skip completely empty rows
                if not sequences_array:
                    continue

                # Complex Naming Architecture Logic
                multimer_prefix = (row.get('multimer_name') or "").strip()
                if multimer_prefix:
                    output_filename = f"{multimer_prefix}__{last_protein_name}.json"
                else:
                    output_filename = f"{last_protein_name}_complex_{row_idx}.json"

                json_data = {
                    "name": output_filename.replace(".json", ""),
                    "sequences": sequences_array,
                    "modelSeeds": list(range(1, 4)),
                    "dialect": "alphafold3",
                    "version": 3
                }

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
