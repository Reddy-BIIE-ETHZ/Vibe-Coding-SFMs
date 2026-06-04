#!/bin/bash -l
# ==============================================================================
# SLURM job: miR-SFM pool-512 evaluation + seed-matching baseline
#
# Runs AFTER euler_eval_mirsfm.slurm completes.
# CPU-only job — no GPU needed.
#
# Usage:  sbatch scripts/euler_pool512_mirsfm.slurm
# ==============================================================================

#SBATCH --job-name=mirsfm-p512
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mirsfm_pool512_%j.out
#SBATCH --error=logs/mirsfm_pool512_%j.err

CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/mirsfm/data"
ROOT_DIR="/cluster/scratch/${USER}/mirsfm/output"

module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CALM_DIR}/src:${PYTHONPATH}"

mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "miR-SFM Pool-512 Evaluation + Baseline"
echo "  Start time: $(date)"
echo "============================================"

cd ${CALM_DIR}

# --- Pool-512 evaluation for each split/fold ---
for split in identity_100 mmseqs_080; do
  EVAL_DIR="${ROOT_DIR}/${split}/mirsfm-${split}/eval"
  if [ -d "${EVAL_DIR}" ]; then
    echo ""
    echo "=== Pool-512: ${split} ==="
    python -m calm.encoder.eval_pool512 \
        --eval_dir "${EVAL_DIR}" \
        --pool_size 512 \
        --n_trials 100
  else
    echo "WARNING: eval dir not found: ${EVAL_DIR}"
  fi
done

# --- Seed-matching baseline ---
echo ""
echo "============================================"
echo "Seed-Matching Baseline"
echo "============================================"

for split in identity_100 mmseqs_080; do
  SPLIT_FILE="${DATA_DIR}/split_index/${split}/split_hash_ids_outerfold_0_innerfold_0.json"
  if [ -f "${SPLIT_FILE}" ]; then
    echo ""
    echo "=== Baseline: ${split} (fold 0) ==="
    python -m calm.encoder.mirsfm_baseline \
        --metadata "${DATA_DIR}/metadata.csv" \
        --split_file "${SPLIT_FILE}" \
        --pool_size 512 \
        --n_trials 100 \
        --output "${ROOT_DIR}/${split}/baseline_results.json"
  fi
done

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "============================================"
