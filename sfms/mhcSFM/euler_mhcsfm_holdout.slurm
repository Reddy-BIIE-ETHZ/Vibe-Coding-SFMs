#!/bin/bash -l
# ==============================================================================
# SLURM job: mhcSFM rare allele holdout validation
# Trains on 91 alleles, evaluates zero-shot retrieval on 14 holdout alleles
# Then runs pool-512 evaluation on the holdout test set
#
# Usage:  sbatch scripts/euler_mhcsfm_holdout.slurm
# ==============================================================================

#SBATCH --job-name=mhcsfm-holdout
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/mhcsfm_holdout_%j.out
#SBATCH --error=logs/mhcsfm_holdout_%j.err

# --- Configuration ---
CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/mhcsfm"
ROOT_DIR="/cluster/scratch/${USER}/mhcsfm/output/mhcsfm_holdout"

# --- Environment setup ---
module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate

export PYTHONUNBUFFERED=1

mkdir -p ${CALM_DIR}/logs
mkdir -p ${ROOT_DIR}

echo "============================================"
echo "mhcSFM Rare Allele Holdout Validation"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Start time: $(date)"
echo "============================================"

cd ${CALM_DIR}

# --- Step 1: Train on 91 alleles (exclude 14 holdout) ---
echo ""
echo "=== STEP 1: Training (91 alleles, 125K train + 31K val) ==="
echo ""

python -m calm.main \
    stage=encoder \
    mode=train \
    run_name=mhcsfm-holdout \
    project_name=mhcsfm-encoder \
    data/db=mhcsfm \
    data/loader=tsfm \
    model/encoder=mhcsfm_ffn \
    train/encoder=mhcsfm_full \
    data.db.data_dir="${DATA_DIR}" \
    data.db.split_method="split_holdout" \
    rootdir="${ROOT_DIR}" \
    fold_list="[0]" \
    wandb=false

echo ""
echo "=== STEP 2: Eval (generate cosine_sim for holdout test set) ==="
echo ""

python -m calm.main \
    stage=encoder \
    mode=eval \
    run_name=mhcsfm-holdout \
    project_name=mhcsfm-encoder \
    data/db=mhcsfm \
    data/loader=tsfm \
    model/encoder=mhcsfm_ffn \
    train/encoder=mhcsfm_full \
    data.db.data_dir="${DATA_DIR}" \
    data.db.split_method="split_holdout" \
    rootdir="${ROOT_DIR}" \
    fold_list="[0]" \
    wandb=false

echo ""
echo "=== STEP 3: Pool-512 evaluation on holdout test set ==="
echo ""

EVAL_DIR="${ROOT_DIR}/mhcsfm-holdout/eval/fold_0"
if [ -f "${EVAL_DIR}/cosine_sim.pt" ]; then
    python -m calm.encoder.eval_pool512_fast \
        --eval_dir "${EVAL_DIR}" \
        --pool_size 512 \
        --n_trials 100 \
        --max_queries 200
    echo "Pool-512 results saved to ${EVAL_DIR}/pool512_results.csv"
else
    echo "ERROR: cosine_sim.pt not found at ${EVAL_DIR}"
fi

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "============================================"
