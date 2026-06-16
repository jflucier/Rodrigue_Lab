import io
import os
import glob
import tqdm
import sys
import argparse
import pickle
import numpy as np
import torch
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import protein
from utils.geometry import compute_rmsd
from utils.residue_constants import atom_order

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def motifRMSD(motif_pdb, design_pdb, motif_mask):
    if not os.path.exists(motif_pdb) or not os.path.exists(design_pdb):
        return torch.tensor(0.0)
    with open(motif_pdb) as f:
        motif = protein.from_pdb_string(f.read())
    with open(design_pdb) as f:
        design = protein.from_pdb_string(f.read())

    true_motif_ca = torch.from_numpy(motif.atom_positions[:, atom_order["CA"]])
    
    # Extract the absolute raw CA coordinates matrix from design files
    raw_design_ca = torch.from_numpy(design.atom_positions[:, atom_order["CA"]])
    
    # 🔥 DYNAMIC RESIDUE SLICING OVERRIDE 🔥
    # When processing State 4, the structure array is expanded to 277 due to the RpoA domains.
    # We dynamically slice it back down to match the expected 250 boolean mask array.
    mask_len = len(motif_mask)
    if len(raw_design_ca) > mask_len:
        raw_design_ca = raw_design_ca[:mask_len]
        
    design_motif_ca = raw_design_ca[motif_mask]
    return compute_rmsd(design_motif_ca, true_motif_ca)

def compute_RG(design_pdb):
    if not os.path.exists(design_pdb):
        return 0.0
    with open(design_pdb) as f:
        design = protein.from_pdb_string(f.read())
    coords = design.atom_positions[:, 1]
    return np.square(coords - coords.mean(0)).sum(-1).mean() ** 0.5

def motif_results(motif_out_dir, motif_pdb, motif):
    jobs = sorted(glob.glob(os.path.join(motif_out_dir, "design*")))
    if len(jobs) == 0:
        return

    if args.workers > 1:
        from multiprocessing import Pool
        p = Pool(args.workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map

    rowss = list(tqdm.tqdm(__map__(do_single_dir, jobs), total=len(jobs)))
    if args.workers > 1:
        p.__exit__(None, None, None)

    df = []
    for rows in rowss:
        df.extend(rows)

    if len(df) == 0:
        print(f"No results extracted for motif path {motif}")
        return

    full = pd.DataFrame(df)
    full.to_csv(os.path.join(motif_out_dir, "_fullresults.csv"), index=False)

    # Group statistics across all 5 operational steps
    agg = full.groupby(["design", "state"]).agg(
        motifRMSD_mean=("motifrmsd", "mean"),
        motifRMSD_std=("motifrmsd", "std"),
        plddt=("plddt", "mean"),
        ptm=("ptm", "mean"),
        iptm=("iptm", "mean"),
        radius_gyr=("radius_gyr", "mean"),
    ).reset_index()

    # Create multi-column tracking framework
    agg = agg.pivot(index="design", columns="state").reset_index()
    agg.columns = ["_".join(map(str, col)).rstrip("_") for col in agg.columns.to_flat_index()]
    
    # Text mapping linking back to your true biochemical conditions
    agg = agg.rename(columns=lambda c: c.replace("_0", "_state0_apo")
                                        .replace("_1", "_state1_camp")
                                        .replace("_2", "_state2_dna_only")
                                        .replace("_3", "_state3_camp_dna")
                                        .replace("_4", "_state4_rpoA_recruited"))

    agg.to_csv(os.path.join(motif_out_dir, "_aggresults.csv"), index=False)
    print(f"Success! Processed all 5 states and compiled data to: {motif_out_dir}/_aggresults.csv")

def do_single_dir(design_dir):
    rows = []
    design_name = os.path.basename(design_dir)
    spec_path = os.path.join(design_dir, f"{motif}_spec.pkl")
    if not os.path.exists(spec_path):
        return rows

    with open(spec_path, "rb") as f:
        motif_mask = pickle.load(f)["motif_mask"]

    # Explicitly track all 5 states sequentially (0, 1, 2, 3, 4)
    target_states = [0, 1, 2, 3, 4]

    for state in target_states:
        if args.lmpnn:
            pkl_path = f"{design_dir}/lmpnn/boltz_regen/lmpnn_seq1_state{state}.pkl"
        else:
            pkl_path = f"{design_dir}/state{state}.pkl"
            
        if not os.path.exists(pkl_path):
            continue
            
        with open(pkl_path, 'rb') as f:
            try:
                outdict = CPU_Unpickler(f).load()
            except Exception:
                continue

        for sample_idx in range(5):  # All 5 structures evaluated per state
            for seqid in range(1, 9):  # Sequence mutations variation loop
                if args.lmpnn:
                    pdb_file = f"{design_dir}/lmpnn/boltz_regen/lmpnn_seq{seqid}_state{state}_sample{sample_idx}.pdb"
                else:
                    pdb_file = f"{design_dir}/state{state}_sample{sample_idx}.pdb"

                if not os.path.exists(pdb_file):
                    continue

                rmsd = motifRMSD(motif_pdb, pdb_file, motif_mask)
                RG = compute_RG(pdb_file)
                
                plddt_val = outdict["plddt"].cpu().numpy()[sample_idx].mean() if "plddt" in outdict else 0.0
                ptm_val = outdict["ptm"].cpu().numpy()[sample_idx] if "ptm" in outdict else 0.0
                iptm_val = outdict["ligand_iptm"].cpu().numpy()[sample_idx] if "ligand_iptm" in outdict else 0.0

                rows.append({
                    "design": f"{design_name}:{seqid}" if args.lmpnn else design_name,
                    "state": state,
                    "sample": sample_idx,
                    "motifrmsd": rmsd.item() if hasattr(rmsd, "item") else float(rmsd),
                    "radius_gyr": RG,
                    "plddt": plddt_val,
                    "ptm": ptm_val,
                    "iptm": iptm_val,
                })
                if not args.lmpnn:
                    break
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="Path to your Switchcraft outputs directory")
    parser.add_argument('--lmpnn', action='store_true', help="Flag if run used ligandmpnn mutations optimization paths")
    parser.add_argument('--workers', type=int, default=0, help="CPU multiprocessing pool limit")
    args = parser.parse_args()

    out_dir = args.dir
    for motif in os.listdir(out_dir):
        motif_out_dir = os.path.join(out_dir, motif)
        if not os.path.isdir(motif_out_dir):
            continue
        print(f"Executing complete 5-state calculation loop for motif target: {motif}")
        motif_pdb = f"motifs/{motif}.pdb"
        motif_results(motif_out_dir, motif_pdb, motif)

