import os

design_pdb = "crp_camp_dna_out/4n9h.5state/design3/state4_sample3.pdb"
source_pdb = "motifs/1lb2.complex.relabelled.pdb"
output_pdb = "crp_camp_dna_out/4n9h.5state/design3/state4_sample3_complete.pdb"


def extract_rpoa_chains(pdb_path):
    rpoa_lines = []
    # Identify and pull all 4 distinct rpoA domains (Chains B, C, E, F)
    target_chains = ["B", "C", "E", "F"]

    if not os.path.exists(pdb_path):
        print(f"Error: Source file {pdb_path} not found.")
        return rpoa_lines

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                chain_id = line[21:22].strip()
                if chain_id in target_chains:
                    rpoa_lines.append(line)
    return rpoa_lines


def build_complete_complex():
    if not os.path.exists(design_pdb):
        print(f"Error: Design file {design_pdb} not found.")
        return

    with open(design_pdb, "r") as f:
        design_lines = f.readlines()

    # Strip out the isolated, trimmed single-helix lines to avoid duplication
    # Standard CRP designs are typically tracked on Chains A and B
    clean_design_lines = []
    for line in design_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            chain_id = line[21:22].strip()
            if chain_id in ["A", "B"]:
                clean_design_lines.append(line)
        elif not line.startswith("END"):
            clean_design_lines.append(line)

    # Grab all 4 native domains in their true spatial orientation
    print("Extracting 4-domain rpoA assembly fields from biological reference...")
    rpoa_lines = extract_rpoa_chains(source_pdb)
    print(f"Found {len(rpoa_lines)} structural rpoA atom lines.")

    # Combine the optimized CRP core with the complete 4-domain framework
    with open(output_pdb, "w") as out:
        out.writelines(
            clean_lines for clean_lines in clean_lines if "CMP" in clean_lines or "dna" in clean_lines or True)
        out.write("TER\n")
        out.writelines(rpoa_lines)
        out.write("END\n")

    print(f"Success! Complete structural recruitment model saved to: {output_pdb}")


if __name__ == "__main__":
    build_complete_complex()
