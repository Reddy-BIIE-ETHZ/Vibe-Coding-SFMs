#!/bin/bash -l
# ==============================================================================
# SLURM job: crisprSFM training with hard negatives — single fold test
#
# Trains identity_100 fold 0 on 60K pairs (30K positive + 30K hard negative).
# Hard negatives have unique IDs so they serve purely as in-batch negatives.
#
# Usage:  sbatch scripts/euler_train_crispr_hn.slurm
# ==============================================================================

#SBATCH --job-name=crispr-hn
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/crispr_hn_train_%j.out
#SBATCH --error=logs/crispr_hn_train_%j.err

CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/crispr/data_hn"
ROOT_DIR="/cluster/scratch/${USER}/crispr/output_hn"

module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

mkdir -p ${CALM_DIR}/logs ${ROOT_DIR}

echo "============================================"
echo "crisprSFM Training + Hard Negatives"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  DATA_DIR:    ${DATA_DIR}"
echo "  ROOT_DIR:    ${ROOT_DIR}"
echo "  Start time:  $(date)"
echo "============================================"

cd ${CALM_DIR}

srun python -m calm.main \
    stage=encoder \
    mode=train \
    run_name=crispr-hn-identity_100-fold0 \
    project_name=crispr-encoder \
    data/db=crispr \
    data/loader=crispr \
    model/encoder=crispr_ffn \
    train/encoder=crispr_full \
    data.db.data_dir="${DATA_DIR}" \
    data.db.split_method="identity_100" \
    rootdir="${ROOT_DIR}" \
    fold_list="[0]" \
    wandb=false

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "============================================"
