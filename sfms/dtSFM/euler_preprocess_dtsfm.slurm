#!/bin/bash -l
# ==============================================================================
# SLURM job: Embed dtSFM drugs (MoLFormer) and proteins (ESM-2) on GPU
#
# This takes the metadata.csv produced by the 'download' step and creates
# embedding tensors for both drugs and kinases.
#
# Estimated time: ~30 min for 2,037 drugs + 429 proteins
#
# Usage:  sbatch scripts/euler_preprocess_dtsfm.slurm
# ==============================================================================

#SBATCH --job-name=dtsfm-embed
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/dtsfm_embed_%j.out
#SBATCH --error=logs/dtsfm_embed_%j.err

# --- Configuration ---
CALM_DIR="${HOME}/CALM-0.1.0"
OUTPUT_DIR="${CALM_DIR}/data/dtsfm"

# --- Environment setup ---
module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate

mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "dtSFM Preprocessing — Drug + Protein Embedding"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CALM_DIR:    ${CALM_DIR}"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  Start time:  $(date)"
echo "============================================"

cd ${CALM_DIR}
export PYTHONUNBUFFERED=1

srun python -m calm.preprocess.dtsfm embed \
    --output_dir ${OUTPUT_DIR} \
    --device cuda \
    --molformer_batch_size 32 \
    --esm_batch_size 2

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "  Output:   ${OUTPUT_DIR}"
echo "============================================"
