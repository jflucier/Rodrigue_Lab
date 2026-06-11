import os

input_file = "/storage/Documents/service/biologie/rodrigue/analysis/20260602_crp_rnapol/1lb2.complex.pdb"
output_file = "/storage/Documents/service/biologie/rodrigue/analysis/20260602_crp_rnapol/1lb2.complex.relabelled.pdb"

with open(input_file, "r") as f:
    lines = f.readlines()

clean_lines = []
seen_atoms = set()

for line in lines:
    if line.startswith("ATOM  ") or line.startswith("HETATM"):
        atom_id = line[6:11].strip()
        chain = line[21:22]

        # Build a unique tracking key for Atom ID + Original Chain Letter
        key = (atom_id, chain)

        # If we have already seen this exact atom ID in this chain, it is the mirrored copy!
        if key in seen_atoms:
            if chain == "B": line = line[:21] + "C" + line[22:]
            if chain == "E": line = line[:21] + "F" + line[22:]
            if chain == "A": line = line[:21] + "D" + line[22:]
        else:
            seen_atoms.add(key)

        # Isolate only your 4 target RpoA tracks (B, E from Copy 1 and C, F from Copy 2)
        final_chain = line[21:22]
        if final_chain in ["B", "C", "E", "F"]:
            is_hetatm = line.startswith("HETATM")

            # CONVERSION LOGIC: Transform non-standard HETATM amino acids into standard ATOM rows
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

            clean_lines.append(line)

os.makedirs("motifs", exist_ok=True)
with open(output_file, "w") as out:
    out.writelines(clean_lines)

print("Success! Fully normalized 4-domain target file saved to: motifs/rpoa_all_4.pdb")
