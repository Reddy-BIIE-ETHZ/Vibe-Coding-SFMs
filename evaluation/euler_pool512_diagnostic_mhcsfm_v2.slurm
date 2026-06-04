#!/bin/bash -l
# ==============================================================================
# DIAGNOSTIC: pool-512 eval on mhcSFM v2 identity_100 fold 0 (Phase 1.6.5)
#
# Purpose: answer "is the in-batch acc plateau real, or is the encoder
# actually learning a representation pool-512 retrieval can detect?"
# Single fold, single split, single checkpoint = fastest possible signal.
#
# Two stages:
#   Stage 1: mode=eval → produces cosine_sim.pt for the val/test split
#   Stage 2: eval_pool512_fast → reads cosine_sim.pt → R@1 + R@5 numbers
#
# Wall: ~30-45 min (eval ~20 min on GPU + pool512 ~5 min on CPU)
#
# Submit:
#   cd ~/CALM-0.1.0
#   sbatch scripts/euler_pool512_diagnostic_mhcsfm_v2.slurm
# ==============================================================================

#SBATCH --job-name=mhcsfm_v2_p512_diag
#SBATCH --output=logs/mhcsfm_v2_pool512_diag_%j.out
#SBATCH --error=logs/mhcsfm_v2_pool512_diag_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g

# --- Modules + venv ---
module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
module load cuda/12.1.1
source $HOME/calm_env/bin/activate

export PYTHONUNBUFFERED=1

CALM_DIR="$HOME/CALM-0.1.0"
SCRATCH_V2="/cluster/scratch/$USER/mhcsfm_v2"
ROOT_DIR="${SCRATCH_V2}/output/full"

SPLIT_METHOD="identity_100"
FOLD=0
RUN_NAME="mhcsfm_v2-${SPLIT_METHOD}-fold${FOLD}"
EVAL_DIR="${ROOT_DIR}/${RUN_NAME}/eval/fold_${FOLD}"

# Audit dest
SUMMARY_DIR="$SFM_SUMMARY_DIR/mhcsfm_v2/$(date +%Y%m%d_%H%M%S)_diagnostic_pool512_${RUN_NAME}"
mkdir -p "$SUMMARY_DIR"
mkdir -p "$CALM_DIR/logs"

echo "============================================"
echo "mhcSFM v2 Phase 1.6.5 — pool-512 diagnostic"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Node:          $(hostname)"
echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Run name:      ${RUN_NAME}"
echo "  Eval dir:      ${EVAL_DIR}"
echo "  Summary dir:   ${SUMMARY_DIR}"
echo "  Start time:    $(date)"
echo "============================================"

cd "$CALM_DIR"

# --- Stage 1: standard eval → produces cosine_sim.pt ---
echo ""
echo "=== Stage 1: mode=eval (writes cosine_sim.pt) ==="
srun python -m calm.main \
    stage=encoder \
    mode=eval \
    run_name=${RUN_NAME} \
    project_name=mhcsfm_v2 \
    data/db=mhcsfm_v2 \
    data/loader=tsfm \
    model/encoder=mhcsfm_v2_ffn \
    train/encoder=mhcsfm_v2_full \
    data.db.data_dir="${SCRATCH_V2}" \
    data.db.split_method="${SPLIT_METHOD}" \
    rootdir="${ROOT_DIR}" \
    fold_list="[${FOLD}]" \
    wandb=false

EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "ERROR: Stage 1 (mode=eval) exited with code $EVAL_EXIT"
    exit $EVAL_EXIT
fi

# --- Sanity: verify cosine_sim.pt was written ---
if [ ! -f "${EVAL_DIR}/cosine_sim.pt" ]; then
    echo "ERROR: cosine_sim.pt not found at ${EVAL_DIR}/"
    ls -la "${EVAL_DIR}/" 2>/dev/null
    exit 3
fi

echo ""
echo "Stage 1 complete. cosine_sim.pt size:"
du -sh "${EVAL_DIR}/cosine_sim.pt"

# --- Stage 2: pool-512 retrieval ---
echo ""
echo "=== Stage 2: eval_pool512_fast ==="
python -m calm.encoder.eval_pool512_fast \
    --eval_dir "${EVAL_DIR}" \
    --pool_size 512 \
    --n_trials 100 \
    --max_queries 200 \
    | tee "${SUMMARY_DIR}/pool512_results.txt"

# --- Audit: copy results + cosine_sim.pt manifest to durable location ---
cp "${EVAL_DIR}"/pool512_results.csv "$SUMMARY_DIR/" 2>/dev/null || true
echo ""
echo "Per-fold results saved to: $SUMMARY_DIR"

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "  Eval dir: ${EVAL_DIR}"
echo "  Summary:  ${SUMMARY_DIR}"
echo "============================================"
