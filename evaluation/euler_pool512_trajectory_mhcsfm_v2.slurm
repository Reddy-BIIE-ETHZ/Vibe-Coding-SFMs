#!/bin/bash -l
# ==============================================================================
# Pool-512 trajectory: identity_100 fold 0 across all 7 val-best ckpts
# Maps R@1 vs epoch to determine whether epoch 9 was the peak or whether the
# val-best gating (`checkpoint_metric: pred_acc`) threw away improving ckpts.
#
# For each epoch in {0, 1, 2, 3, 5, 8, 9}:
#   1. Copy ckpt to an isolated diag dir (so original ckpts untouched)
#   2. Run mode=eval against identity_100_diag5k split (small cosine matrix)
#   3. Run eval_pool512_fast → R@1 + R@5 + R@10
#   4. Append to summary CSV
#
# Wall: ~7 epochs × (~3min eval + 1min pool512) = ~30-40 min total
#
# Submit:
#   cd ~/CALM-0.1.0
#   sbatch scripts/euler_pool512_trajectory_mhcsfm_v2.slurm
# ==============================================================================

#SBATCH --job-name=mhcsfm_v2_p512_traj
#SBATCH --output=logs/mhcsfm_v2_pool512_traj_%j.out
#SBATCH --error=logs/mhcsfm_v2_pool512_traj_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g

module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
module load cuda/12.1.1
source $HOME/calm_env/bin/activate
export PYTHONUNBUFFERED=1

CALM_DIR="$HOME/CALM-0.1.0"
SCRATCH_V2="/cluster/scratch/$USER/mhcsfm_v2"
SOURCE_CKPT_DIR="${SCRATCH_V2}/output/full/mhcsfm_v2-identity_100-fold0/train/fold_0"
DIAG_ROOT="${SCRATCH_V2}/diag_traj"
SUMMARY_DIR="$SFM_SUMMARY_DIR/mhcsfm_v2/$(date +%Y%m%d_%H%M%S)_pool512_trajectory_identity_100_fold0"
SUMMARY_CSV="${SUMMARY_DIR}/trajectory_results.csv"
SUMMARY_TXT="${SUMMARY_DIR}/trajectory_log.txt"

mkdir -p "$SUMMARY_DIR" "$CALM_DIR/logs"

# Epochs we have val-best ckpts for (from `ls best_model_val_pred_acc_epoch_*.pth`)
EPOCHS=(0 1 2 3 5 8 9)

echo "============================================"
echo "mhcSFM v2 pool-512 trajectory: identity_100 fold 0"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Node:          $(hostname)"
echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Source ckpts:  ${SOURCE_CKPT_DIR}"
echo "  Diag root:     ${DIAG_ROOT}"
echo "  Summary dir:   ${SUMMARY_DIR}"
echo "  Epochs:        ${EPOCHS[@]}"
echo "  Start time:    $(date)"
echo "============================================"

# Initialize summary CSV with header
echo "epoch,R1_ag2ab,R5_ag2ab,R10_ag2ab,R1_ab2ag,R5_ab2ag,R10_ab2ag" > "$SUMMARY_CSV"

cd "$CALM_DIR"

for EPOCH in "${EPOCHS[@]}"; do
    echo ""
    echo "============================================"
    echo "=== Epoch ${EPOCH} ==="
    echo "============================================"

    SRC_CKPT="${SOURCE_CKPT_DIR}/best_model_val_pred_acc_epoch_${EPOCH}.pth"
    if [ ! -f "$SRC_CKPT" ]; then
        echo "  WARNING: source ckpt not found: $SRC_CKPT"
        echo "${EPOCH},MISSING,MISSING,MISSING,MISSING,MISSING,MISSING" >> "$SUMMARY_CSV"
        continue
    fi

    # Set up isolated diag dir for this epoch
    DIAG_RUN_NAME="diag_epoch_${EPOCH}"
    DIAG_TRAIN_DIR="${DIAG_ROOT}/${DIAG_RUN_NAME}/train/fold_0"
    DIAG_EVAL_DIR="${DIAG_ROOT}/${DIAG_RUN_NAME}/eval/fold_0"
    mkdir -p "$DIAG_TRAIN_DIR"

    # Copy ckpt under a name the auto-discovery picks (highest-epoch best_model)
    cp "$SRC_CKPT" "${DIAG_TRAIN_DIR}/best_model_val_pred_acc_epoch_${EPOCH}.pth"
    echo "  Copied ${SRC_CKPT} -> ${DIAG_TRAIN_DIR}/"

    # Run mode=eval against the diag5k split (small cosine matrix)
    echo ""
    echo "  --- Stage 1: mode=eval (5K diag split) ---"
    srun python -m calm.main \
        stage=encoder mode=eval \
        run_name=${DIAG_RUN_NAME} project_name=mhcsfm_v2 \
        data/db=mhcsfm_v2 data/loader=tsfm \
        model/encoder=mhcsfm_v2_ffn train/encoder=mhcsfm_v2_full \
        data.db.data_dir="${SCRATCH_V2}" \
        data.db.split_method="identity_100_diag5k" \
        rootdir="${DIAG_ROOT}" \
        fold_list="[0]" \
        wandb=false 2>&1 | tail -15

    if [ ! -f "${DIAG_EVAL_DIR}/cosine_sim.pt" ]; then
        echo "  ERROR: cosine_sim.pt not produced for epoch ${EPOCH}"
        echo "${EPOCH},EVAL_FAIL,EVAL_FAIL,EVAL_FAIL,EVAL_FAIL,EVAL_FAIL,EVAL_FAIL" >> "$SUMMARY_CSV"
        continue
    fi

    # Run pool-512 fast and capture output
    echo ""
    echo "  --- Stage 2: pool-512 fast ---"
    POOL_OUTPUT=$(python -m calm.encoder.eval_pool512_fast \
        --eval_dir "${DIAG_EVAL_DIR}" \
        --pool_size 512 \
        --n_trials 100 \
        --max_queries 200 2>&1)

    echo "$POOL_OUTPUT" | tail -20

    # Parse R@k values from the summary block. Lines look like:
    #   R@1_ag2ab: 7.1 ± 0.0
    #   R@1_ab2ag: 74.4 ± 0.0
    R1_AG=$(echo "$POOL_OUTPUT" | grep "R@1_ag2ab" | awk '{print $2}')
    R5_AG=$(echo "$POOL_OUTPUT" | grep "R@5_ag2ab" | awk '{print $2}')
    R10_AG=$(echo "$POOL_OUTPUT" | grep "R@10_ag2ab" | awk '{print $2}')
    R1_AB=$(echo "$POOL_OUTPUT" | grep "R@1_ab2ag" | awk '{print $2}')
    R5_AB=$(echo "$POOL_OUTPUT" | grep "R@5_ab2ag" | awk '{print $2}')
    R10_AB=$(echo "$POOL_OUTPUT" | grep "R@10_ab2ag" | awk '{print $2}')

    echo "${EPOCH},${R1_AG:-NA},${R5_AG:-NA},${R10_AG:-NA},${R1_AB:-NA},${R5_AB:-NA},${R10_AB:-NA}" >> "$SUMMARY_CSV"
    echo ""
    echo "  Logged: epoch ${EPOCH}: R@1 ag2ab=${R1_AG}%  ab2ag=${R1_AB}%"
done

echo ""
echo "============================================"
echo "=== TRAJECTORY COMPLETE ==="
echo "============================================"
echo ""
echo "Summary CSV:"
cat "$SUMMARY_CSV"
echo ""
echo "Files saved to: $SUMMARY_DIR"
echo "End time: $(date)"
