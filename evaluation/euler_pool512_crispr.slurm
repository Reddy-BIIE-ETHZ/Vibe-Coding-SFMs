#!/bin/bash -l
# ==============================================================================
# SLURM job: crisprSFM pool-512 scoring — NO GPU needed
#
# Runs fast pool-512 evaluation on all cosine_sim.pt files.
# Subsamples 200 queries per fold for tractability (6000+ test set is too
# large for exhaustive evaluation).
#
# Usage:  sbatch scripts/euler_pool512_crispr.slurm
# ==============================================================================

#SBATCH --job-name=crispr-p512
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/crispr_pool512_%j.out
#SBATCH --error=logs/crispr_pool512_%j.err

# --- Configuration ---
CALM_DIR="${HOME}/CALM-0.1.0"
ROOT_DIR="/cluster/scratch/${USER}/crispr/output"

# --- Environment setup ---
module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate
export PYTHONUNBUFFERED=1

echo "============================================"
echo "crisprSFM Pool-512 Evaluation (fast)"
echo "  Start time: $(date)"
echo "============================================"

cd ${CALM_DIR}

for SPLIT in identity_100 hamming_080 hamming_060 hamming_040; do
    # Check if eval dirs exist
    if [ ! -d "${ROOT_DIR}/crispr-${SPLIT}-fold0/eval" ]; then
        echo "Skipping ${SPLIT} — eval directory not found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "=== ${SPLIT} ==="
    echo "=========================================="

    # Symlink all fold eval dirs into one parent
    COMBINED_EVAL="${ROOT_DIR}/pool512_eval/${SPLIT}"
    mkdir -p ${COMBINED_EVAL}

    for FOLD in 0 1 2 3 4; do
        SRC="${ROOT_DIR}/crispr-${SPLIT}-fold${FOLD}/eval/fold_${FOLD}"
        if [ -d "${SRC}" ]; then
            ln -sf ${SRC} ${COMBINED_EVAL}/fold_${FOLD}
        fi
    done

    python -m calm.encoder.eval_pool512_fast \
        --eval_dir ${COMBINED_EVAL} \
        --pool_size 512 \
        --n_trials 100 \
        --max_queries 200 \
        --output ${ROOT_DIR}/pool512_results_${SPLIT}.csv
done

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "  Results:"
ls -lh ${ROOT_DIR}/pool512_results_*.csv 2>/dev/null
echo "============================================"
