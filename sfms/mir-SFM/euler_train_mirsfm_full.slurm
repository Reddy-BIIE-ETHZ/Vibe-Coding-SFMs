#!/bin/bash -l
# ==============================================================================
# SLURM array job: miR-SFM full training (4 OOD thresholds × 5 folds = 20 runs)
#
# Submit AFTER euler_mmseqs_mirsfm.slurm has completed.
# Each array task trains one (split_method, fold) combination.
#
# Usage:  sbatch scripts/euler_train_mirsfm_full.slurm
# ==============================================================================

#SBATCH --job-name=mirsfm-full
#SBATCH --array=0-19
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/mirsfm_full_%A_%a.out
#SBATCH --error=logs/mirsfm_full_%A_%a.err

# --- Configuration ---
CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/mirsfm/data"
ROOT_DIR="/cluster/scratch/${USER}/mirsfm/output"

# --- Map array task ID to (split_method, fold) ---
# 4 split methods × 5 folds = 20 tasks (IDs 0-19)
SPLIT_METHODS=("identity_100" "mmseqs_080" "mmseqs_060" "mmseqs_040")
SPLIT_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
SPLIT_METHOD="${SPLIT_METHODS[$SPLIT_IDX]}"

# --- Environment ---
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CALM_DIR}/src:${PYTHONPATH}"

mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "miR-SFM Full Training"
echo "  Job ID:      ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID})"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Split:       ${SPLIT_METHOD}"
echo "  Fold:        ${FOLD}"
echo "  DATA_DIR:    ${DATA_DIR}"
echo "  ROOT_DIR:    ${ROOT_DIR}/${SPLIT_METHOD}"
echo "  Start time:  $(date)"
echo "============================================"

cd ${CALM_DIR}

srun python -m calm.main \
    stage=encoder mode=train \
    run_name="mirsfm-${SPLIT_METHOD}" \
    project_name=mirsfm-encoder \
    data/db=mirsfm \
    data/loader=mirsfm \
    model/encoder=mirsfm_ffn \
    train/encoder=mirsfm_full \
    data.db.data_dir="${DATA_DIR}" \
    data.db.split_method="${SPLIT_METHOD}" \
    rootdir="${ROOT_DIR}/${SPLIT_METHOD}" \
    fold_list="[${FOLD}]" \
    wandb=false

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "============================================"
