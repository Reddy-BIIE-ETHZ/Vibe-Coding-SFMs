#!/bin/bash -l
# ==============================================================================
# SLURM array job: Full mhcSFM training
# 4 OOD thresholds × 5 folds = 20 runs
#
# Usage:  sbatch scripts/euler_train_mhcsfm_full.slurm
#
# This submits 20 parallel GPU jobs. Each takes ~2-4 hours (168K pairs).
# Array index 0-19 maps to (split_method, fold):
#   0-4:   identity_100, folds 0-4
#   5-9:   mmseqs_080,   folds 0-4
#   10-14: mmseqs_060,   folds 0-4
#   15-19: mmseqs_040,   folds 0-4
# ==============================================================================

#SBATCH --job-name=mhcsfm-full
#SBATCH --array=0-19
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/mhcsfm_full_%A_%a.out
#SBATCH --error=logs/mhcsfm_full_%A_%a.err

# --- Configuration ---
CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/mhcsfm"
ROOT_DIR="/cluster/scratch/${USER}/mhcsfm/output/mhcsfm_full"

# --- Map array index to (split_method, fold) ---
SPLIT_METHODS=("identity_100" "mmseqs_080" "mmseqs_060" "mmseqs_040")
SPLIT_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
SPLIT_METHOD=${SPLIT_METHODS[$SPLIT_IDX]}

# --- Environment setup ---
module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate

mkdir -p ${CALM_DIR}/logs
mkdir -p ${ROOT_DIR}

export PYTHONUNBUFFERED=1

echo "============================================"
echo "mhcSFM Full Training"
echo "  Array Job ID:  ${SLURM_ARRAY_JOB_ID}"
echo "  Task ID:       ${SLURM_ARRAY_TASK_ID}"
echo "  Node:          $(hostname)"
echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  SPLIT_METHOD:  ${SPLIT_METHOD}"
echo "  FOLD:          ${FOLD}"
echo "  Start time:    $(date)"
echo "============================================"

cd ${CALM_DIR}

srun python -m calm.main \
    stage=encoder \
    mode=train \
    run_name=mhcsfm-${SPLIT_METHOD}-fold${FOLD} \
    project_name=mhcsfm-encoder \
    data/db=mhcsfm \
    data/loader=tsfm \
    model/encoder=mhcsfm_ffn \
    train/encoder=mhcsfm_full \
    data.db.data_dir="${DATA_DIR}" \
    data.db.split_method="${SPLIT_METHOD}" \
    rootdir="${ROOT_DIR}" \
    fold_list="[${FOLD}]" \
    wandb=false

echo ""
echo "============================================"
echo "  End time:      $(date)"
echo "  Split method:  ${SPLIT_METHOD}"
echo "  Fold:          ${FOLD}"
echo "  Output:        ${ROOT_DIR}/mhcsfm-${SPLIT_METHOD}-fold${FOLD}"
echo "============================================"
