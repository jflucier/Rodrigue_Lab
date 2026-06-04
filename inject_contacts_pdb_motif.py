import json
import os


def prepare_contact_perfect_motif(pdb_name):
    pdb_path = f"motifs/{pdb_name}.pdb"
    if not os.path.exists(pdb_path):
        print(f"Error: {pdb_path} not found.")
        return

    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    # Clear out any old tracking headers
    clean_lines = [l for l in lines if not l.startswith("REMARK    ") or "motif_spec" not in l]

    # Exactly map the four contact clusters for Chain A and Chain B
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
        "name": pdb_name.lower(),
        "segments": segments
    }

    header = f"REMARK    motif_spec {json.dumps(spec)}\n"

    with open(pdb_path, 'w') as f:
        f.write(header)
        f.writelines(clean_lines)
    print(f"Successfully injected exact contact segments into {pdb_path}")


# Apply to your templates
prepare_contact_perfect_motif("4N9H")
prepare_contact_perfect_motif("2GZW")
