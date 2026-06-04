#!/bin/bash -l
# ==============================================================================
# SLURM job: Embed CRISPR off-target data for crisprSFM
#
# Runs DNABERT-2 embedding on guide RNA and off-target DNA sequences.
# Data must already be downloaded (metadata.csv + cclmoff_raw.csv cached)
# from Mac preprocessing. This job only does the embedding step.
#
# IMPORTANT: GPU nodes have NO internet. Download DNABERT-2 model weights
# on the login node first:
#   python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M'); AutoModel.from_pretrained('zhihan1996/DNABERT-2-117M')"
#
# Usage:  sbatch scripts/euler_preprocess_crispr.slurm
# ==============================================================================

#SBATCH --job-name=crispr-embed
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/crispr_embed_%j.out
#SBATCH --error=logs/crispr_embed_%j.err

# --- Configuration ---
CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/crispr/data"

# --- Environment setup ---
module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "crisprSFM Embedding — DNABERT-2"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CALM_DIR:    ${CALM_DIR}"
echo "  DATA_DIR:    ${DATA_DIR}"
echo "  Start time:  $(date)"
echo "============================================"

cd ${CALM_DIR}

# Run preprocessing with cached data + exclusion file
# CCLMoff is already cached as cclmoff_raw.csv — no download needed
python -m calm.preprocess.crispr \
    --output_dir ${DATA_DIR} \
    --use_cclmoff \
    --exclude_guides_file ${DATA_DIR}/exclude_crisprofft_guides.txt \
    --device cuda \
    --dna_batch_size 16

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "  Output files:"
ls -lh ${DATA_DIR}/*.pt ${DATA_DIR}/metadata.csv 2>/dev/null
echo "============================================"
