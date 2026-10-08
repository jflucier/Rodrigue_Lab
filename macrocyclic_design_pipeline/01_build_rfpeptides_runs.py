#!/usr/bin/env python
"""
01_build_rfpeptides_runs.py

Reads a TSV of targets (design_name, pdb_path, hotspots[, length][, target_pos])
and writes one runnable Slurm sbatch script per row (.slurm).

Optional column target_pos: one contiguous residue range of the target chain
to design against, in the input PDB's numbering ("1-66" or "A1-66"). It
replaces the whole-chain range in the RFdiffusion contig, so the rest of the
target is simply not given to the model. Hotspots keep the PDB's own
numbering and must all lie inside target_pos. Empty/absent = whole chain.

Each design directory records its contig/hotspot parameters in
run_params.json. If outputs already exist there from a run with DIFFERENT
parameters (e.g. you added a crop), the script stops instead of mixing them;
use a new design_name or delete the old outputs.

Automatically inspects target folders to see which 6-hour chunks (tasks)
have already completed all their PDB outputs, creating a custom sparse
Slurm array parameter (e.g., --array=1,3-5) to completely avoid overwriting
and safely pick up unfinished work.
"""
import argparse
import csv
import json
import sys
import math
import re
from pathlib import Path

try:
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.pdbx as pdbx_io
except ImportError:
    sys.exit("This script needs biotite: pip install biotite")

# Adjusted template to dynamically inject a custom sparse array instruction string
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=RFdiff_{design_name}
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --array={array_range_str}
{cluster_specific_headers}#SBATCH --output={abs_out_dir}/{design_name}/logs/slurm-%A_%a.out

set -euo pipefail

# Calculate chunk slices based on the Slurm Array Task ID
DESIGNS_PER_TASK={designs_per_task}
TOTAL_TARGET_DESIGNS={total_designs}

START_NUM=$(( SLURM_ARRAY_TASK_ID * DESIGNS_PER_TASK ))

# The final array task handles whatever leftover count remains
REMAINING_DESIGNS=$(( TOTAL_TARGET_DESIGNS - START_NUM ))
if [ $REMAINING_DESIGNS -lt $DESIGNS_PER_TASK ]; then
    NUM_DESIGNS=$REMAINING_DESIGNS
else
    NUM_DESIGNS=$DESIGNS_PER_TASK
fi

echo "=========================================================="
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Targeting Backbone Block Range: $START_NUM to $(( START_NUM + NUM_DESIGNS - 1 ))"
echo "Generating $NUM_DESIGNS designs on Node: $SLURMD_NODENAME"
echo "=========================================================="

# Split this task's designs across WORKERS concurrent RFdiffusion processes that
# share the one GPU (each design is small, so a single process leaves it mostly idle).
WORKERS={workers}
BASE=$(( NUM_DESIGNS / WORKERS ))
EXTRA=$(( NUM_DESIGNS % WORKERS ))
OFFSET=$START_NUM
PIDS=""

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

for W in $(seq 0 $(( WORKERS - 1 ))); do
    N=$BASE
    if [ $W -lt $EXTRA ]; then N=$(( N + 1 )); fi
    if [ $N -le 0 ]; then continue; fi
    echo "worker $W: $N designs starting at index $OFFSET"
    singularity exec --nv \\
        -B /home/jflucier/programs/RFdiffusion:/home/jflucier/programs/RFdiffusion \\
        -B /home/jflucier/programs/RFdiffusion/schedules:/opt/RFdiffusion/schedules \\
        -B {abs_in_dir}:{abs_in_dir} \\
        -B {abs_out_dir}:{abs_out_dir} \\
        -B $CUDA_MPS_PIPE_DIRECTORY \\
        {container_sif} \\
        python3 /opt/RFdiffusion/scripts/run_inference.py \\
        --config-name base \\
        inference.output_prefix={output_prefix} \\
        inference.num_designs=$N \\
        inference.design_startnum=$OFFSET \\
        'contigmap.contigs=[{length} {chain}{lo}-{hi}/0]' \\
        inference.input_pdb={pdb_path} \\
        inference.cyclic=True \\
        diffuser.T=50 \\
        inference.cyc_chains='a' \\
        'ppi.hotspot_res=[{hotspot_list}]' \\
        hydra.run.dir={hydra_log_dir}/task_${{SLURM_ARRAY_TASK_ID}}_w${{W}} \\
        hydra.output_subdir={hydra_log_dir}/task_${{SLURM_ARRAY_TASK_ID}}_w${{W}}/hydra \\
        > {hydra_log_dir}/worker_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}_w${{W}}.log 2>&1 &
    PIDS="$PIDS $!"
    OFFSET=$(( OFFSET + N ))
done

FAIL=0
for P in $PIDS; do
    wait $P || FAIL=1
done
if [ $FAIL -ne 0 ]; then
    echo "At least one RFdiffusion worker failed; see worker_*.log in {hydra_log_dir}"
    exit 1
fi
"""


def load_structure(pdb_path: str):
    p = Path(pdb_path)
    if p.suffix.lower() in (".cif", ".mmcif"):
        cif = pdbx_io.CIFFile.read(str(p))
        return pdbx_io.get_structure(cif, model=1)
    f = pdb_io.PDBFile.read(str(p))
    return pdb_io.get_structure(f, model=1)


def chain_residue_range(structure, chain_id: str):
    mask = structure.chain_id == chain_id
    if not mask.any():
        raise ValueError(f"Chain '{chain_id}' not found in structure.")
    res_ids = structure.res_id[mask]
    return int(res_ids.min()), int(res_ids.max())


def parse_hotspots(hotspot_str: str):
    tokens = [t.strip() for t in hotspot_str.split(",") if t.strip()]
    chains = set()
    for tok in tokens:
        if not tok:
            continue
        chains.add(tok[0])
    if len(chains) != 1:
        raise ValueError(
            f"All hotspots in one row must be on the same chain, got: {hotspot_str}"
        )
    return chains.pop(), tokens


def resolve_target_pos(target_pos: str, chain: str, full_lo: int, full_hi: int, hotspot_tokens):
    """Parse target_pos and validate it. Returns (lo, hi)."""
    m = re.fullmatch(r"([A-Za-z]?)(\d+)\s*-\s*(\d+)", target_pos.strip())
    if not m:
        raise ValueError(
            f"target_pos {target_pos!r} must be a single range like '1-66' or 'A1-66'")
    tp_chain, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
    if tp_chain and tp_chain != chain:
        raise ValueError(f"target_pos chain {tp_chain} differs from hotspot chain {chain}")
    if lo > hi:
        raise ValueError(f"target_pos {lo}-{hi} is reversed")
    if lo < full_lo or hi > full_hi:
        raise ValueError(
            f"target_pos {lo}-{hi} is outside chain {chain} residues {full_lo}-{full_hi}")
    try:
        positions = [int(t[1:]) for t in hotspot_tokens]
    except ValueError:
        raise ValueError(f"cannot parse hotspot residue numbers in {hotspot_tokens}")
    outside = [t for t, n in zip(hotspot_tokens, positions) if not lo <= n <= hi]
    if outside:
        raise ValueError(f"hotspots outside target_pos {lo}-{hi}: {', '.join(outside)}")
    return lo, hi


def check_run_params(design_out_dir: Path, params: dict, name: str):
    """Refuse to continue a design directory that was generated with other parameters."""
    params_file = design_out_dir / "run_params.json"
    has_outputs = any(design_out_dir.glob("diffused_binder_cyclic_*.pdb"))
    if params_file.exists():
        old = json.loads(params_file.read_text())
        if old != params:
            diff = {k: (old.get(k), params.get(k)) for k in set(old) | set(params)
                    if old.get(k) != params.get(k)}
            sys.exit(f"[{name}] existing outputs in {design_out_dir} were generated with "
                     f"different parameters (old, new): {diff}\n"
                     f"Use a new design_name or delete the old outputs.")
    elif has_outputs:
        print(f"[{name}] WARNING: {design_out_dir} already holds designs but no run_params.json; "
              f"cannot verify they match the current contig/hotspots. Not recording parameters.")
        return
    params_file.write_text(json.dumps(params, indent=2) + "\n")


def find_missing_tasks(design_out_dir: Path, total_designs: int, designs_per_task: int) -> list:
    """Scans for existing PDBs and returns a list of task IDs that are incomplete."""
    if not design_out_dir.exists():
        total_tasks = math.ceil(total_designs / designs_per_task)
        return list(range(total_tasks))

    # Read all generated file indexes
    pattern = re.compile(r"diffused_binder_cyclic_(\d+)\.pdb$")
    existing_indices = set()
    for p_file in design_out_dir.glob("diffused_binder_cyclic_*.pdb"):
        match = pattern.search(p_file.name)
        if match:
            existing_indices.add(int(match.group(1)))

    total_tasks = math.ceil(total_designs / designs_per_task)
    missing_tasks = []

    for task_id in range(total_tasks):
        start_idx = task_id * designs_per_task
        # Calculate exactly how many items this block was assigned
        end_idx = min(start_idx + designs_per_task, total_designs)

        # Check if every single file index in this slice exists
        task_slice_indices = set(range(start_idx, end_idx))
        if not task_slice_indices.issubset(existing_indices):
            missing_tasks.append(task_id)

    return missing_tasks


def build_array_string(tasks: list) -> str:
    """Converts a sorted list of task integers into a condensed Slurm range string.
    e.g. [0, 1, 2, 4, 6, 7] -> "0-2,4,6-7"
    """
    if not tasks:
        return ""
    ranges = []
    start = tasks[0]
    end = tasks[0]

    for t in tasks[1:]:
        if t == end + 1:
            end = t
        else:
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")
            start = t
            end = t
    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{end}")

    return ",".join(ranges)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets_tsv")
    ap.add_argument("--out-dir", default="rfpeptides_runs",
                    help="Directory to write per-target run scripts + outputs into")
    ap.add_argument("--account", default="def-rodrigu1",
                    help="Slurm allocation account name charge group identifier")
    ap.add_argument("--container-sif",
                    default="/home/jflucier/programs/Rodrigue_Lab/macrocyclic_design_pipeline/containers/rfdiffusion_gh200.sif",
                    help="Path to the Apptainer/Singularity container file (.sif)")
    ap.add_argument("--default-length", default="5-15",
                    help="Macrocycle length used when a row has no 'length' column")
    ap.add_argument("--num-designs", type=int, default=10000,
                    help="Total targeted number of backbones desired per run profile")
    ap.add_argument("--designs-per-job", type=int, default=600,
                    help="Number of designs generated per 6-hour window allocation slice")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent RFdiffusion processes sharing the one GPU inside each array task "
                         "(each design uses only ~2 GB of GPU memory). Default 1 = previous behavior.")
    ap.add_argument("--time", default="06:00:00", help="Slurm time limit per array task")
    ap.add_argument("--cpus-per-task", type=int, default=None,
                    help="Default: max(4, 2 x workers)")
    ap.add_argument("--mem", default=None, help="Default: max(32, 8 x workers) GB")
    ap.add_argument("--cluster", default="", choices=["", "gh"],
                    help="Target cluster profiling configuration ruleset selection")
    ap.add_argument("--queue", default="",
                    help="The partition destination queue required when --cluster=gh is set")
    args = ap.parse_args()

    if args.workers < 1:
        sys.exit("--workers must be >= 1")
    cpus = args.cpus_per_task or max(4, 2 * args.workers)
    mem = args.mem or f"{max(32, 8 * args.workers)}G"

    # Process cluster specific modifications
    clusterHeaders = ""
    if args.cluster == "gh":
        queue_name = args.queue
        if not queue_name:
            queue_name = "gh-preempt-low"
            print("\n" + "!" * 72)
            print("WARNING: No --queue specified for cluster 'gh'.")
            print("         Defaulting to partition: 'gh-preempt-low'.")
            print("         Your array jobs are subject to preemption on this queue.")
            print("!" * 72 + "\n")

        clusterHeaders = f"#SBATCH -p {queue_name}\n#SBATCH --requeue\n"


    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.targets_tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"design_name", "pdb_path", "hotspots"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"TSV missing required columns: {missing}")

        written = []
        for row in reader:
            name = row["design_name"].strip()
            if not name:
                continue
            pdb_path = Path(row["pdb_path"].strip()).resolve()
            hotspots_str = row["hotspots"].strip()
            length = (row.get("length") or "").strip() or args.default_length

            try:
                structure = load_structure(str(pdb_path))
                chain, hotspot_tokens = parse_hotspots(hotspots_str)
                lo, hi = chain_residue_range(structure, chain)
                full_lo, full_hi = lo, hi
                target_pos = (row.get("target_pos") or "").strip()
                if target_pos:
                    lo, hi = resolve_target_pos(target_pos, chain, full_lo, full_hi,
                                                hotspot_tokens)
            except Exception as e:
                sys.exit(f"[{name}] failed to process: {e}")

            hotspot_list = ",".join(hotspot_tokens)
            design_out_dir = out_dir / name
            design_out_dir.mkdir(exist_ok=True)
            print(f"[{name}] target {chain}{lo}-{hi} ({hi - lo + 1} of {full_hi - full_lo + 1} "
                  f"residues); hotspots: {hotspot_list}")
            check_run_params(design_out_dir, {
                "pdb_path": str(pdb_path), "length": length,
                "target": f"{chain}{lo}-{hi}", "hotspots": hotspot_list}, name)

            hydra_log_dir = design_out_dir / "logs"
            hydra_log_dir.mkdir(exist_ok=True)

            # Analyze files on disk to find which block segments are incomplete
            missing_tasks = find_missing_tasks(design_out_dir, args.num_designs, args.designs_per_job)

            if not missing_tasks:
                print(
                    f"[{name}] All tasks have completely generated. Target threshold fully reached. Skipping script setup.")
                continue

            array_range_str = build_array_string(missing_tasks)
            print(f"[{name}] Incomplete/Missing tasks detected. Creating Slurm execution array for: {array_range_str}")

            script_text = SLURM_TEMPLATE.format(
                design_name=name,
                account=args.account,
                time=args.time, cpus=cpus, mem=mem, workers=args.workers,
                container_sif=args.container_sif,
                abs_in_dir=str(pdb_path.parent),
                abs_out_dir=str(out_dir),
                output_prefix=str(design_out_dir / "diffused_binder_cyclic"),
                total_designs=args.num_designs,
                designs_per_task=args.designs_per_job,
                array_range_str=array_range_str,
                cluster_specific_headers=clusterHeaders,
                length=length,
                chain=chain, lo=lo, hi=hi,
                pdb_path=str(pdb_path),
                hotspot_list=hotspot_list,
                hydra_log_dir=str(hydra_log_dir)
            )

            script_path = out_dir / f"run_{name}.slurm"
            script_path.write_text(script_text)
            script_path.chmod(0o755)
            written.append(script_path)

    print(f"\nWrote {len(written)} Slurm sbatch script(s) to {out_dir}/")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
