#!/bin/bash -l
# ==============================================================================
# SLURM job: Embed eSFM substrates (MoLFormer) and enzymes (ESM-2) on GPU
#
# This takes the metadata.csv produced by the 'download' step and creates
# embedding tensors for both substrates and enzymes.
#
# Estimated time:
#   MoLFormer (2,646 substrates): ~5 min
#   ESM-2 (177,389 enzymes):      ~15-20 hours on A100
#   Total:                         ~20 hours
#
# Output goes to /cluster/scratch/ because the ESM-2 embeddings for 177K
# enzymes will be VERY LARGE (~50-100 GB).
#
# Usage:  sbatch scripts/euler_preprocess_esfm.slurm
# ==============================================================================

#SBATCH --job-name=esfm-embed
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --output=logs/esfm_embed_%j.out
#SBATCH --error=logs/esfm_embed_%j.err

# --- Configuration ---
CALM_DIR="${HOME}/CALM-0.1.0"
# Use scratch for large embeddings (home quota = 50GB, not enough)
OUTPUT_DIR="/cluster/scratch/${USER}/esfm"

# --- Environment setup ---
module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate

mkdir -p ${CALM_DIR}/logs
mkdir -p ${OUTPUT_DIR}

# Copy metadata.csv to scratch if not already there
if [ ! -f "${OUTPUT_DIR}/metadata.csv" ]; then
    echo "Copying metadata.csv to scratch..."
    cp ${CALM_DIR}/data/esfm/metadata.csv ${OUTPUT_DIR}/
fi

echo "============================================"
echo "eSFM Preprocessing — Substrate + Enzyme Embedding"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CALM_DIR:    ${CALM_DIR}"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  Start time:  $(date)"
echo "============================================"

cd ${CALM_DIR}
export PYTHONPATH=src:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

python -m calm.preprocess.esfm embed \
    --output_dir ${OUTPUT_DIR} \
    --device cuda \
    --molformer_batch_size 32 \
    --esm_batch_size 2

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "  Output:   ${OUTPUT_DIR}"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Copy split files: cp -r ${CALM_DIR}/data/esfm/split_index ${OUTPUT_DIR}/"
echo "  2. Run MMseqs2: sbatch scripts/euler_mmseqs_esfm.slurm"
echo "  3. Smoketest: sbatch scripts/euler_smoketest_esfm.slurm"
