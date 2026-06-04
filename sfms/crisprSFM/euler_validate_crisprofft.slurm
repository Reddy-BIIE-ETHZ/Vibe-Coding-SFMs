#!/bin/bash -l
# ==============================================================================
# SLURM job: CRISPRoffT pseudo-prospective validation for crisprSFM
#
# Embeds held-out CRISPRoffT sequences, projects through trained model,
# evaluates Tier 1 (standard retrieval) and Tier 2 (hard-negative
# discrimination), and compares against Hamming distance baseline.
#
# Uses the identity_100 fold 0 model (trained on all CCLMoff minus
# overlapping CRISPRoffT guides).
#
# Usage:  sbatch scripts/euler_validate_crisprofft.slurm
# ==============================================================================

#SBATCH --job-name=crispr-val
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/crispr_validate_%j.out
#SBATCH --error=logs/crispr_validate_%j.err

# --- Configuration ---
CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/crispr/data"
ROOT_DIR="/cluster/scratch/${USER}/crispr/output"

# Use identity_100 fold 0 model (retrained on train+val)
MODEL_CKPT="${ROOT_DIR}/crispr-identity_100-fold0/train/fold_0/final_model_epoch_99.pth"
VALIDATION_CSV="${DATA_DIR}/crisprofft_validation_full.csv"

# --- Environment setup ---
module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "CRISPRoffT Validation Experiment"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Model:       ${MODEL_CKPT}"
echo "  Validation:  ${VALIDATION_CSV}"
echo "  Start time:  $(date)"
echo "============================================"

cd ${CALM_DIR}

python -m calm.encoder.validate_crisprofft \
    --validation_csv ${VALIDATION_CSV} \
    --model_checkpoint ${MODEL_CKPT} \
    --model_config_dir ${CALM_DIR}/configs \
    --device cuda \
    --batch_size 16 \
    --n_trials 100 \
    --output ${ROOT_DIR}/crisprofft_validation_results.json

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "============================================"
