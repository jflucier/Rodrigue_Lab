import os
import argparse
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Superimposer


def get_ca_atoms(chain):
    """Extract all CA atoms from a given chain configuration."""
    atoms = []
    for residue in chain:
        if "CA" in residue:
            atoms.append(residue["CA"])
    return atoms


def main():
    parser = argparse.ArgumentParser(
        description="Perfectly align and stitch 5-state multi-chain structures based on exact ChimeraX tracks."
    )
    parser.add_argument("-d", "--design", required=True, help="Path to Switchcraft state4_sample3.pdb file")
    parser.add_argument("-r", "--reference", required=True, help="Path to 1lb2.complex.relabelled.pdb reference")
    parser.add_argument("-o", "--output", required=True, help="Path to save the complete composite PDB file")

    args = parser.parse_args()

    if not os.path.exists(args.design) or not os.path.exists(args.reference):
        print("Error: Input files not found. Check your paths.")
        return

    parser_pdb = PDBParser(QUIET=True)

    # Load structures into memory
    design_struct = parser_pdb.get_structure("design", args.design)
    ref_struct = parser_pdb.get_structure("ref", args.reference)

    design_model = design_struct[0]
    ref_model = ref_struct[0]

    # 1. Gather alignment targets for the central CRP template core
    # Design protein core lives exclusively on Chain A
    design_ca = get_ca_atoms(design_model["A"])

    # Reference protein core typically lives on Chains A and B
    ref_ca = []
    for c_id in ["A", "B"]:
        if c_id in ref_model:
            ref_ca.extend(get_ca_atoms(ref_model[c_id]))

    # Match lengths cleanly to avoid shape mismatch boundaries
    n_match = min(len(design_ca), len(ref_ca))
    if n_match == 0:
        print("Error: Could not isolate Alpha Carbon coordinates for structural alignment.")
        return

    print(f"Aligning frames using {n_match} structural matching residues...")

    # 2. Calculate the exact 3D spatial rotation and translation vectors
    superimposer = Superimposer()
    superimposer.set_atoms(design_ca[:n_match], ref_ca[:n_match])

    # 3. Transform the reference assembly coordinate system
    # This brings the full 4-domain framework into the design space
    print("Transforming reference coordinate space...")
    rot, tran = superimposer.rotran  # 🔥 BUGFIX: Unpack the rotation and translation tuple

    for chain in ref_model:
        for residue in chain:
            for atom in residue:
                coord = atom.get_coord()
                # Apply rotation matrix multiplication, then add translation vector offset
                new_coord = np.dot(coord, rot) + tran
                atom.set_coord(new_coord)

    # 4. Assemble the composite model geometry
    output_structure = Structure.Structure("complete_complex")
    output_model = Model.Model(0)
    output_structure.add(output_model)

    # RETAIN ALL DESIGN ASSETS: Keep the purple dimer (A) and the four DNA strands (D, E, F, G)
    # This completely discards the truncated single helix (H)
    target_design_chains = ["A", "D", "E", "F", "G"]
    for chain_id in target_design_chains:
        if chain_id in design_model:
            chain = design_model[chain_id]
            new_chain = Chain.Chain(chain.id)
            for res in chain:
                new_chain.add(res.copy())
            output_model.add(new_chain)

    # APPEND THE RE-ALIGNED RPOA DOMAINS (B, C, E, F)
    # Renaming to unique tracking letters (K, L, M, N) prevents any chain ID crashes with DNA
    rpoa_map = {"B": "K", "C": "L", "E": "M", "F": "N"}
    for old_id, new_id in rpoa_map.items():
        if old_id in ref_model:
            chain = ref_model[old_id]
            new_chain = Chain.Chain(new_id)
            for res in chain:
                # Update residue and atom parent chain tracking properties on the fly
                res_copy = res.copy()
                new_chain.add(res_copy)
            output_model.add(new_chain)

    # Save out the composite file
    io = PDBIO()
    io.set_structure(output_structure)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    io.save(args.output)
    print(f"Success! Complete composite structural model saved to: {args.output}")


if __name__ == "__main__":
    main()
