#!/bin/bash -l
# ==============================================================================
# SLURM array job: Evaluate tSFM — generate cosine_sim.pt for pool-512 eval
# 4 OOD thresholds × 5 folds = 20 evaluation runs
#
# Usage:  sbatch scripts/euler_eval_tsfm.slurm
# ==============================================================================

#SBATCH --job-name=tsfm-eval
#SBATCH --array=0-19
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/tsfm_eval_%A_%a.out
#SBATCH --error=logs/tsfm_eval_%A_%a.err

# --- Configuration ---
CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/tsfm/jaspar_data_v2_50x"
ROOT_DIR="/cluster/scratch/${USER}/tsfm/output/tsfm_full"

# --- Map array index to (split_method, fold) ---
SPLIT_METHODS=("identity_100" "mmseqs_080" "mmseqs_060" "mmseqs_040")
SPLIT_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
SPLIT_METHOD=${SPLIT_METHODS[$SPLIT_IDX]}
RUN_NAME="tsfm-${SPLIT_METHOD}-fold${FOLD}"

# --- Environment setup ---
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate

export PYTHONUNBUFFERED=1

mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "tSFM Evaluation"
echo "  Task ID:       ${SLURM_ARRAY_TASK_ID}"
echo "  SPLIT_METHOD:  ${SPLIT_METHOD}"
echo "  FOLD:          ${FOLD}"
echo "  RUN_NAME:      ${RUN_NAME}"
echo "  Start time:    $(date)"
echo "============================================"

cd ${CALM_DIR}

srun python -m calm.main \
    stage=encoder \
    mode=eval \
    run_name=${RUN_NAME} \
    project_name=tsfm-encoder \
    data/db=jaspar \
    data/loader=tsfm \
    model/encoder=tsfm_ffn \
    train/encoder=tsfm_full \
    data.db.data_dir="${DATA_DIR}" \
    data.db.split_method="${SPLIT_METHOD}" \
    rootdir="${ROOT_DIR}" \
    fold_list="[${FOLD}]" \
    wandb=false

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "============================================"
