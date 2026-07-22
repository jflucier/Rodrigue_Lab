#!/bin/bash
#SBATCH --job-name=RFdiff_binder
#SBATCH --output=RFdiff_binder_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
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
LEN_RANGE="15-80"
NUM_DESIGNS=200
RAW_HOTSPOTS=""
PROGRAMS=/scratch/jflucier/programs/RFdiffusion
CONTAINER_SIF="/scratch/jflucier/programs/proteinmpnn.sif"

TEMP=$(getopt -o n:b:c:l:d:p:s --long name:,base_path:,chain:,len_range:,num_designs:,hotspots:,skip_rfdiffusion -n 'run_rfdiffusion.slurm' -- "$@")
if [ $? -ne 0 ]; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"

while true ; do
    case "$1" in
        -n|--name) NAME="$2" ; shift 2 ;;
        -b|--base_path) BASE_PATH="$2" ; shift 2 ;;
        -c|--chain) CHAIN="$2" ; shift 2 ;;
        -l|--len_range) LEN_RANGE="$2" ; shift 2 ;;
        -d|--num_designs) NUM_DESIGNS="$2" ; shift 2 ;;
        -p|--hotspots) RAW_HOTSPOTS="$2" ; shift 2 ;;
        -s|--skip_rfdiffusion) SKIP_RF=1 ; shift ;;
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

if [ ! -f "${CONTAINER_SIF}" ]; then
    echo "Error: The ProteinMPNN Singularity container image was not found!"
    echo "Expected path: ${CONTAINER_SIF}"
    exit 1
fi

INPUT_PDB="${BASE_PATH}/pdb/${NAME}.pdb"
RF_OUT="${BASE_PATH}/binders/${NAME}"
MPNN_OUT="${BASE_PATH}/binders/${NAME}/mpnn_results"

if [ ! -f "${INPUT_PDB}" ]; then
    echo "Error: The target PDB structure file does not exist!"
    echo "Expected path: ${INPUT_PDB}"
    exit 1
fi

echo "Analyzing ${INPUT_PDB} for chain ${CHAIN} sequence gaps..."
# Convert comma-separated string (e.g., "A,B") into a bash space-separated array
IFS=',' read -r -a ATOM_CHAINS <<< "$CHAIN"

CONTIG_STR=""
FIXED_CHAINS_LIST=""

for CURRENT_CH in "${ATOM_CHAINS[@]}"; do
    # Strip any accidental whitespace
    CURRENT_CH=$(echo "$CURRENT_CH" | tr -d ' ')

    # Extract sorted, unique residue numbers specifically for this loop's chain
    RESIDUES=($(awk -v ch="$CURRENT_CH" 'BEGIN{FS=""} /^ATOM/ && $22==ch {print substr($0,23,4)}' "$INPUT_PDB" | sort -nu))

    if [ ${#RESIDUES[@]} -eq 0 ]; then
        echo "Error: No residues found for chain ${CURRENT_CH} in PDB file."
        exit 1
    fi

    START_SEG=${RESIDUES[0]}
    PREV_SEG=${RESIDUES[0]}

    # Loop over residues to discover structural gaps within the current chain
    for RES in "${RESIDUES[@]:1}"; do
        if [ $((RES - PREV_SEG)) -gt 1 ]; then
            CONTIG_STR="${CONTIG_STR}${CURRENT_CH}${START_SEG}-${PREV_SEG}/0 "
            START_SEG=$RES
        fi
        PREV_SEG=$RES
    done

    CONTIG_STR="${CONTIG_STR}${CURRENT_CH}${START_SEG}-${PREV_SEG}/0 "
    FIXED_CHAINS_LIST="${FIXED_CHAINS_LIST}${CURRENT_CH}"
done

CONTIG_STR=$(echo "$CONTIG_STR" | tr -s ' ' | sed 's/[[:space:]]*$//')
FIXED_CHAINS_LIST=$(echo "$FIXED_CHAINS_LIST" | tr -s ' ' | sed 's/[[:space:]]*$//')

FULL_CONTIG="[${CONTIG_STR} ${LEN_RANGE}]"
echo "Generated Gap-Aware Multi-Chain Contig: ${FULL_CONTIG}"
echo "Identified Fixed Target Structural Chains: [${FIXED_CHAINS_LIST}]"
echo "Requested number of generation designs: ${NUM_DESIGNS}"

HOTSPOTS_ARG=""
if [ -n "$RAW_HOTSPOTS" ]; then
    # Strip any accidental spaces and wrap in brackets
    CLEAN_HOTSPOTS=$(echo "$RAW_HOTSPOTS" | tr -d ' ')
    HOTSPOTS_ARG="ppi.hotspot_res=[${CLEAN_HOTSPOTS}]"
    echo "Configured RFdiffusion Hotspots: ${HOTSPOTS_ARG}"
fi

WDIR="${RF_OUT}/work"
CACHE_DIR="${RF_OUT}/schedules_cache"
mkdir -p "$WDIR" "$RF_OUT" "$MPNN_OUT" "$CACHE_DIR"

if [ "$SKIP_RF" -eq 1 ]; then
    echo "Notice: --skip_rfdiffusion parameter passed."
    # Ensure that target directory has structures to pass down to ProteinMPNN
    if [ -z "$(ls -A "$RF_OUT" 2>/dev/null | grep '\.pdb$')" ]; then
        echo "Error: Cannot skip RFdiffusion because no pre-existing generated PDB files were found in $RF_OUT"
        exit 1
    fi
    echo "Skipping RFdiffusion generation. Using existing backbones inside: $RF_OUT"
else
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
  ${HOTSPOTS_ARG} \
  denoiser.noise_scale_ca=0 \
  denoiser.noise_scale_frame=0
fi

echo "running parse_multiple_chains"
apptainer exec --nv --bind /scratch/jflucier:/scratch/jflucier "$CONTAINER_SIF" \
    python /app/proteinmpnn/helper_scripts/parse_multiple_chains.py \
    --input_path="$RF_OUT" \
    --output_path="$MPNN_OUT/parsed_pdbs.jsonl"

# Safely isolate the alphabetically last input chain to compute the design letter assignment
LAST_INPUT_CHAIN=$(echo "${ATOM_CHAINS[@]}" | tr ' ' '\n' | sort | tail -n 1)
DESIGN_CHAIN=$(echo "$LAST_INPUT_CHAIN" | tr 'A-Za-z' 'B-ZAb-za')

echo "Target chain(s) are [${CHAIN}]. Setting ProteinMPNN to design Binder Chain: ${DESIGN_CHAIN}"
echo "running assign_fixed_chains"
apptainer exec --nv --bind /scratch/jflucier:/scratch/jflucier "$CONTAINER_SIF" \
    python /app/proteinmpnn/helper_scripts/assign_fixed_chains.py \
    --input_path="$MPNN_OUT/parsed_pdbs.jsonl" \
    --output_path="$MPNN_OUT/chain_id_map.jsonl" \
    --chain_list "${DESIGN_CHAIN}"

# 2. Run ProteinMPNN
echo "running protein_mpnn_run"
apptainer exec --nv --bind /scratch/jflucier:/scratch/jflucier "$CONTAINER_SIF" \
    python /app/proteinmpnn/protein_mpnn_run.py \
    --jsonl_path "$MPNN_OUT/parsed_pdbs.jsonl" \
    --chain_id_jsonl "$MPNN_OUT/chain_id_map.jsonl" \
    --out_folder "$MPNN_OUT" \
    --num_seq_per_target 10 \
    --sampling_temp "0.2" \
    --batch_size 1 --seed 0

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

tail -n +2 ${BASE_PATH}/binders/${NAME}_mpnnmetrics.unsorted.tsv | \
sort -t$'\t' -k2,2n | \
awk -F'\t' '!seen[$6]++' >> ${BASE_PATH}/binders/${NAME}_mpnnmetrics.tsv

rm ${BASE_PATH}/binders/${NAME}_mpnnmetrics.unsorted.tsv

echo "done"