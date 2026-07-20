#!/bin/bash
#SBATCH --job-name=RFdiff_binder
#SBATCH --output=RFdiff_binder_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62G
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --time=48:00:00
#SBATCH --account=def-rodrigu1

ml apptainer/1.3.5
ml cuda/12.2
source /scratch/jflucier/programs/RFdiffusion/venv/bin/activate

#export NVIDIA_TF32_OVERRIDE=0
#export TORCH_CUDA_ARCH_LIST="8.0"
#export HYDRA_FULL_ERROR=1

NAME=""
#/scratch/jflucier/20260708_binders_alexis
BASE_PATH=""
CHAIN=""
PROGRAMS=/scratch/jflucier/programs/RFdiffusion
CONTAINER_SIF="/scratch/jflucier/programs/proteinmpnn.sif"

TEMP=$(getopt -o n:b:c: --long name:,base_path:,chain: -n 'run_rfdiffusion.slurm' -- "$@")
if [ $? -ne 0 ]; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"

while true ; do
    case "$1" in
        -n|--name) NAME="$2" ; shift 2 ;;
        -b|--base_path) BASE_PATH="$2" ; shift 2 ;;
        -c|--chain) CHAIN="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Invalid Option" ; exit 1 ;;
    esac
done

# Validate that the name parameter was actually passed
if [ -z "$NAME" ]; then
    echo "Error: The -n or --name option is required. Must ba a pdb found in ${BASE_PATH}/pdb/${NAME}.pdb"
    echo "Usage: sbatch run_rfdiffusion.slurm --name '1ETN'"
    exit 1
fi

# Fallback Logic: If base_path is empty, use current working directory ($PWD)
if [ -z "$BASE_PATH" ]; then
    BASE_PATH="$PWD"
    echo "Notice: No base_path provided. Defaulting to current directory: ${BASE_PATH}"
fi

INPUT_PDB="${BASE_PATH}/pdb/${NAME}.pdb"
RF_OUT="${BASE_PATH}/binders/${NAME}"
MPNN_OUT="${BASE_PATH}/binders/${NAME}/mpnn_results"
NUM_DESIGNS=200

if [ ! -f "${INPUT_PDB}" ]; then
    echo "Error: The target PDB structure file does not exist!"
    echo "Expected path: ${INPUT_PDB}"
    exit 1
fi

echo "Analyzing ${INPUT_PDB} for chain ${CHAIN} sequence gaps..."

# Extract sorted, unique residue numbers for the chosen chain
RESIDUES=($(awk -v ch="$CHAIN" 'BEGIN{FS=""} /^ATOM/ && $22==ch {print substr($0,23,4)}' "$INPUT_PDB" | sort -nu))

if [ ${#RESIDUES[@]} -eq 0 ]; then
    echo "Error: No residues found for chain ${CHAIN} in PDB file."
    exit 1
fi

CONTIG_STR=""
START_SEG=${RESIDUES[0]}
PREV_SEG=${RESIDUES[0]}

for RES in "${RESIDUES[@]:1}"; do
    # Check if there is a jump/gap in numbering
    if [ $((RES - PREV_SEG)) -gt 1 ]; then
        # Close the completed continuous segment and append it
        CONTIG_STR="${CONTIG_STR}${CHAIN}${START_SEG}-${PREV_SEG}/0 "
        START_SEG=$RES
    fi
    PREV_SEG=$RES
done
# Append the final remaining structural segment
CONTIG_STR="${CONTIG_STR}${CHAIN}${START_SEG}-${PREV_SEG}/0"

# Append your binder generation definition (15 to 80 amino acids long)
FULL_CONTIG="[${CONTIG_STR} 15-80]"
echo "Generated Gap-Aware Contig: ${FULL_CONTIG}"

#if [ -z "$CHAIN" ]; then
#    # Grab the first character found in column 5 (the chain ID column) of the first ATOM row
#    CHAIN=$(grep -m 1 "^ATOM" "$INPUT_PDB" | awk '{print substr($0, 22, 1)}')
#    echo "Notice: No explicit chain provided. Automatically detected Target Chain: ${CHAIN}"
#else
#    # Verify the user-provided chain letter actually exists in the file
#    CHAIN_CHECK=$(grep "^ATOM" "$INPUT_PDB" | awk '{print substr($0, 22, 1)}' | grep -m 1 "$CHAIN")
#    if [ -z "$CHAIN_CHECK" ]; then
#        echo "Error: The requested chain '${CHAIN}' was not found in ${INPUT_PDB}!"
#        exit 1
#    fi
#    echo "Success: Using explicitly requested Target Chain: ${CHAIN}"
#fi

if [ ! -f "${CONTAINER_SIF}" ]; then
    echo "Error: The ProteinMPNN Singularity container image was not found!"
    echo "Expected path: ${CONTAINER_SIF}"
    exit 1
fi

WDIR="${RF_OUT}/work"
CACHE_DIR="${RF_OUT}/schedules_cache"
mkdir -p "$WDIR" "$RF_OUT" "$MPNN_OUT" "$CACHE_DIR"

# Define hotspots from BOTH chains at the interface (e.g., A10,B25)
#HOTSPOTS="[A498,A499,B336,B306]"

# Extract the exact first and last residue numbers from the PDB file dynamically
FIRST_RES=$(awk -v ch="$CHAIN" 'BEGIN{FS=""} /^ATOM/ && $22==ch {print substr($0,23,4); exit}' "$INPUT_PDB" | tr -d ' ')
LAST_RES=$(awk -v ch="$CHAIN" 'BEGIN{FS=""} /^ATOM/ && $22==ch {last=substr($0,23,4)} END{print last}' "$INPUT_PDB" | tr -d ' ')
echo "Detected residue range for Target Chain ${CHAIN}: ${FIRST_RES} to ${LAST_RES}"

# Run RFdiffusion to target the interface
echo "running rfdiffusion"
#  If hotspots, add 'ppi.hotspot_res='$HOTSPOTS \
python ${PROGRAMS}/scripts/run_inference.py \
inference.input_pdb="$INPUT_PDB" \
inference.output_prefix="$RF_OUT/${NAME}" \
inference.num_designs="$NUM_DESIGNS" \
inference.model_directory_path=${PROGRAMS}/models \
hydra.run.dir="$WDIR" \
"contigmap.contigs=${FULL_CONTIG}" \
denoiser.noise_scale_ca=0 \
denoiser.noise_scale_frame=0

echo "running parse_multiple_chains"
apptainer exec --nv --bind /scratch/jflucier:/scratch/jflucier "$CONTAINER_SIF" \
    python /app/proteinmpnn/helper_scripts/parse_multiple_chains.py \
    --input_path="$RF_OUT" \
    --output_path="$MPNN_OUT/parsed_pdbs.jsonl"

echo "running assign_fixed_chains"
apptainer exec --nv --bind /scratch/jflucier:/scratch/jflucier "$CONTAINER_SIF" \
    python /app/proteinmpnn/helper_scripts/assign_fixed_chains.py \
    --input_path="$MPNN_OUT/parsed_pdbs.jsonl" \
    --output_path="$MPNN_OUT/chain_id_map.jsonl" \
    --chain_list "A"

# 2. Run ProteinMPNN
echo "running protein_mpnn_run"
apptainer exec --nv --bind /scratch/jflucier:/scratch/jflucier "$CONTAINER_SIF" \
    python /app/proteinmpnn/protein_mpnn_run.py \
    --jsonl_path "$MPNN_OUT/parsed_pdbs.jsonl" \
    --chain_id_jsonl "$MPNN_OUT/chain_id_map.jsonl" \
    --out_folder "$MPNN_OUT" \
    --num_seq_per_target 10 \
    --sampling_temp "0.2" \
    --batch_size 1

echo "Sequence design complete. Compiling and sorting metrics table..."
awk '
BEGIN {
    FS = "[, \t]+";
    OFS = "\t";
    print "binder_name", "score", "global_score", "fixed_chains", "designed_chains", "sequence"
}
# Match the primary backbone reference header
/^>'${NAME}'_[0-9]+/ {
    gsub(/>/, "", $1);
    current_backbone = $1;

    # Track metadata elements across varying field locations
    for (i = 2; i <= NF; i++) {
        if ($i ~ /^score=/)          { split($i, val, "="); b_score = val[2] }
        if ($i ~ /^global_score=/)   { split($i, val, "="); b_global = val[2] }
        if ($i ~ /^fixed_chains=/)   { split($i, val, "="); gsub(/[][]|\047/, "", val[2]); b_fixed = val[2] }
        if ($i ~ /^designed_chains=/) { split($i, val, "="); gsub(/[][]|\047/, "", val[2]); b_designed = val[2] }
    }
    next
}
# Match individual sequence sample iterations
/^>T=/ {
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^sample=/)       { split($i, val, "="); current_sample = val[2] }
        if ($i ~ /^score=/)        { split($i, val, "="); s_score = val[2] }
        if ($i ~ /^global_score=/) { split($i, val, "="); s_global = val[2] }
    }
    getline_flag = 1;
    next
}
# Capture sequence and print structured metrics
getline_flag == 1 {
    print current_backbone "_sample_" current_sample, s_score, s_global, b_fixed, b_designed, $0;
    getline_flag = 0
}' ${BASE_PATH}/binders/${NAME}/mpnn_results/seqs/*.fa > ${BASE_PATH}/binders/${NAME}_mpnnmetrics.unsorted.tsv

# Lock down the header row, sort the numeric body data, and compile everything into your final file
head -n 1 ${BASE_PATH}/binders/${NAME}_mpnnmetrics.unsorted.tsv > ${BASE_PATH}/binders/${NAME}_mpnnmetrics.tsv
tail -n +2 ${BASE_PATH}/binders/${NAME}_mpnnmetrics.unsorted.tsv | sort -t$'\t' -k2,2n >> ${BASE_PATH}/binders/${NAME}_mpnnmetrics.tsv
rm ${BASE_PATH}/binders/${NAME}_mpnnmetrics.unsorted.tsv

echo "done"