#!/bin/bash
#SBATCH --job-name=afast
#SBATCH --output=afast_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62G
#SBATCH --gpus=h100:1
#SBATCH --time=48:00:00
#SBATCH --account=def-rodrigu1

ml apptainer/1.3.5
ml cuda/12.2

NAME=""
BASE_PATH="/scratch/jflucier/20260602_crp_rnapol/alphafast"
AFAST="/scratch/jflucier/programs/alphafast"

echo "running alphafast on ${BASE_PATH}/inputs"
${AFAST}/scripts/run_alphafast.sh \
    --input_dir ${BASE_PATH}/inputs \
    --output_dir ${BASE_PATH}/outputs \
    --db_dir ${AFAST}/dbs \
    --weights_dir ${AFAST}/weights \
    --temp_dir $SLURM_TMPDIR/alphafast_tmp \
    --container ${AFAST}/alphafast.sif \
    --jax_compilation_cache_dir $SLURM_TMPDIR/alphafast_jax_cache \
    --gpu_devices 0
echo "done"