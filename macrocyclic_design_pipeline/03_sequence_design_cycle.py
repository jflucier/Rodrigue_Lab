#!/usr/bin/env python
"""
03_sequence_design_cycle.py

Generates an optimized SLURM batch script to run the
ProteinMPNN -> PyRosetta iterative macrocycle pipeline on a GPU node,
and prints out the terminal command required to run it.
Uses a single combined container file holding both software stacks.
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--proteinmpnn-sif", required=True,
                    help="Path to the combined SIF file containing both ProteinMPNN and PyRosetta")
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--max-backbones", type=int, default=None,
                    help="Maximum number of backbone structures to process (Default: process all)")
    ap.add_argument("--queue", default="gh-bio",
                    help="The partition destination queue")
    ap.add_argument("--bind-path", default="/net/nfs-ip34",
                    help="Absolute host path to bind mount into the container (Default: /net/nfs-ip34)")
    args = ap.parse_args()

    # Resolve absolute physical paths to prevent cluster symlink errors
    backbones_path = Path(args.backbones_dir).resolve()
    out_path = Path(args.out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "logs").mkdir(exist_ok=True)

    # Locate backbones
    pdbs = sorted(backbones_path.glob("*.pdb")) + sorted(backbones_path.glob("*.cif"))
    if not pdbs:
        sys.exit(f"No .pdb/.cif files found under {backbones_path}")

    # Slice the input backbone array if a maximum limit parameter is active
    if args.max_backbones is not None:
        pdbs = pdbs[:args.max_backbones]
        print(f"Capping design generation queue to the first {len(pdbs)} backbone structures.")

    # Build internal path variables
    slurm_script_path = out_path / "run_pipeline_job.slurm"
    fast_relax_xml = Path(__file__).resolve().parent / "rosetta" / "fast_relax_cyclize.xml"

    # =============================================================================
    # GENERATE THE .SLURM SCRIPT HEADERS (Using standard f-string for args injection)
    # =============================================================================
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=macrocycle_pipeline
#SBATCH --partition={args.queue}
#SBATCH --output={out_path}/logs/pipeline_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

echo "=== Starting Iterative Macrocycle Pipeline ==="
echo "Running on node: \$(hostname)"

# Internal container paths
MPNN_SCRIPT="/opt/proteinmpnn/protein_mpnn_run.py"
MPNN_WEIGHTS="/opt/proteinmpnn/vanilla_model_weights/v_48_020.pt"
"""

    # =============================================================================
    # DYNAMICALLY INJECT BACKBONE LOOPS (Using safe string replacement instead of f-string)
    # =============================================================================
    for pdb in pdbs:
        pdb_absolute = pdb.resolve()
        stem = pdb.stem

        # Raw block completely prevents Python from parsing or consuming backslashes or variables
        backbone_template = r"""
# -----------------------------------------------------------------------------
# Backbone: __STEM__
# -----------------------------------------------------------------------------
CURRENT_PDB="__PDB_ABSOLUTE__"

for rnd in $(seq 1 __N_ROUNDS__); do
    ROUND_DIR="__OUT_PATH__/__STEM__/round${rnd}"
    mkdir -p "${ROUND_DIR}"

    echo "[__STEM__] === Round ${rnd} === Running ProteinMPNN"
    singularity exec --nv --pwd /tmp -B __BIND_PATH__ \
        __MPNN_SIF__ \
        python3 "${MPNN_SCRIPT}" \
        --pdb_path "${CURRENT_PDB}" \
        --pdb_path_chains "A" \
        --temperature "0.0001" \
        --backbone_noise "0" \
        --omit_AAs "C" \
        --num_seq_per_target 1 \
        --path_to_model_weights "${MPNN_WEIGHTS}" \
        --out_folder "${ROUND_DIR}"

    MPNN_OUT=$(find "${ROUND_DIR}/seqs" -name "*.fa" | head -n 1)
    if [ -z "${MPNN_OUT}" ]; then
        echo "[WARN] No ProteinMPNN output found for __STEM__ round ${rnd}, breaking chain loops."
        break
    fi

    echo "[__STEM__] === Round ${rnd} === Running PyRosetta FastRelax"
    RELAXED_PDB="${ROUND_DIR}/__STEM___r${rnd}.pdb"
    TMP_SCRIPT="${ROUND_DIR}/__STEM___r${rnd}.relax.py"

    # Write out separate python runtime script
    cat << 'EOF' > "${TMP_SCRIPT}"
from pyrosetta import *
init('-beta_nov16 -mute all')

xml = '__XML_PATH__'
objs = protocols.rosetta_scripts.XmlObjects.create_from_file(xml)
fr = objs.get_mover('full_relax_complex')
pcm = objs.get_mover('pcm')

pose = pose_from_pdb('${CURRENT_PDB}')
pcm.apply(pose)
fr.apply(pose)
pcm.apply(pose)
pose.dump_pdb('${RELAXED_PDB}')
EOF

    # Run PyRosetta within container
    singularity exec --nv --pwd /tmp -B __BIND_PATH__ \
        __MPNN_SIF__ \
        python3 "${TMP_SCRIPT}"

    # Advance pointer state
    CURRENT_PDB="${RELAXED_PDB}"
done
"""
        # Precise text replacement maps parameters into place with no escaping anomalies
        processed_block = backbone_template.replace("__STEM__", str(stem))
        processed_block = processed_block.replace("__PDB_ABSOLUTE__", str(pdb_absolute))
        processed_block = processed_block.replace("__N_ROUNDS__", str(args.n_rounds))
        processed_block = processed_block.replace("__OUT_PATH__", str(out_path))
        processed_block = processed_block.replace("__BIND_PATH__", str(args.bind_path))
        processed_block = processed_block.replace("__MPNN_SIF__", str(args.proteinmpnn_sif))
        processed_block = processed_block.replace("__XML_PATH__", str(fast_relax_xml.resolve()))

        slurm_content += processed_block

    # Save finalized text to disk
    slurm_script_path.write_text(slurm_content)

    # =============================================================================
    # OUTPUT TRACES OF COMMANDS TO EXECUTE
    # =============================================================================
    print("\n" + "=" * 80)
    print(" SLURM BATCH SCRIPT GENERATION COMPLETE")
    print("=" * 80)
    print(f"File saved to: {slurm_script_path}")
    print(f"Target Queue: {args.queue}")
    print(f"Total Backbones Scaled into Job: {len(pdbs)}")
    print("\nTo submit this job to your cluster queue, execute the following command:")
    print(f"sbatch {slurm_script_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
