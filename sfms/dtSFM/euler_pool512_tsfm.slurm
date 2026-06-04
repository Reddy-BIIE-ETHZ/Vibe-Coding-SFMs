#!/bin/bash -l
# ==============================================================================
# SLURM job: tSFM pool-512 evaluation with durable summary CSVs
#
# Runs AFTER euler_eval_tsfm.slurm completes.
# CPU-only job — no GPU needed.
#
# Writes per-split summary CSVs to a durable location (home dir, NOT scratch)
# via the SFM_SUMMARY_DIR env var picked up by eval_pool512.py.
# Per audit/PRESERVATION_DISCIPLINE.md Rule 2.
#
# Usage:  sbatch scripts/euler_pool512_tsfm.slurm
#
# Note on directory layout: tSFM's training pipeline creates one run-dir per
# (split, fold) — i.e., tsfm-${split}-fold${FOLD}/eval/fold_${FOLD}/. This
# script iterates over both split and fold so every fold contributes a row to
# the durable per-split summary CSV.
# ==============================================================================

#SBATCH --job-name=tsfm-p512
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/tsfm_pool512_%j.out
#SBATCH --error=logs/tsfm_pool512_%j.err

CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/tsfm/jaspar_data_v2_50x"
ROOT_DIR="/cluster/scratch/${USER}/tsfm/output/tsfm_full"

module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CALM_DIR}/src:${PYTHONPATH}"

# Durable summary location — survives scratch purges (Rule 2).
export SFM_SUMMARY_DIR="${HOME}/CALM-0.1.0/eval_summaries"
mkdir -p "${SFM_SUMMARY_DIR}/tsfm"
mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "tSFM Pool-512 Evaluation"
echo "  ROOT_DIR:        ${ROOT_DIR}"
echo "  SFM_SUMMARY_DIR: ${SFM_SUMMARY_DIR}"
echo "  Start time:      $(date)"
echo "============================================"

cd ${CALM_DIR}

# --- Pool-512 evaluation for each split, all folds ---
# Each (split, fold) lives in its own run directory. We pass --output to
# get one CSV per split that aggregates all five folds (rows). The fallback
# durable resolution (SFM_SUMMARY_DIR + sfm_name + split_name) is also set
# so the env-driven path is exercised when the array layout changes.
for split in identity_100 mmseqs_080 mmseqs_060 mmseqs_040; do
  echo ""
  echo "============================================"
  echo "=== Pool-512: ${split} ==="
  echo "============================================"

  SPLIT_OUTPUT="${SFM_SUMMARY_DIR}/tsfm/${split}_pool512_results.csv"
  SPLIT_HEADER_WRITTEN=0
  : > "${SPLIT_OUTPUT}.tmp"

  for fold in 0 1 2 3 4; do
    EVAL_DIR="${ROOT_DIR}/tsfm-${split}-fold${fold}/eval"
    FOLD_CSV="${EVAL_DIR}/pool512_fold${fold}.csv"

    if [ ! -d "${EVAL_DIR}" ]; then
      echo "WARNING: eval dir not found: ${EVAL_DIR} (skipping)"
      continue
    fi

    echo ""
    echo "--- ${split} fold ${fold} ---"
    # Use the fast (subsampled-query) variant. The full eval_pool512.py is
    # O(N_test x n_trials) with single-threaded numpy and runs ~hours per
    # fold at the 50x scale. eval_pool512_fast.py subsamples to 200 random
    # queries and finishes in ~30-60s with metrics within sampling noise.
    python -m calm.encoder.eval_pool512_fast \
        --eval_dir "${EVAL_DIR}" \
        --pool_size 512 \
        --n_trials 100 \
        --max_queries 200 \
        --sfm_name tsfm \
        --split_name "${split}_fold${fold}" \
        --output "${FOLD_CSV}"

    if [ -f "${FOLD_CSV}" ]; then
      if [ "${SPLIT_HEADER_WRITTEN}" -eq 0 ]; then
        head -n 1 "${FOLD_CSV}" >> "${SPLIT_OUTPUT}.tmp"
        SPLIT_HEADER_WRITTEN=1
      fi
      tail -n +2 "${FOLD_CSV}" >> "${SPLIT_OUTPUT}.tmp"
    fi
  done

  if [ -s "${SPLIT_OUTPUT}.tmp" ]; then
    mv "${SPLIT_OUTPUT}.tmp" "${SPLIT_OUTPUT}"
    echo ""
    echo "  Aggregated split CSV: ${SPLIT_OUTPUT}"
    wc -l "${SPLIT_OUTPUT}"
  else
    rm -f "${SPLIT_OUTPUT}.tmp"
    echo "  WARNING: no fold CSVs produced for split ${split}"
  fi
done

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "  Durable summaries: ${SFM_SUMMARY_DIR}/tsfm/"
ls -la "${SFM_SUMMARY_DIR}/tsfm/" 2>/dev/null || true
echo "============================================"
