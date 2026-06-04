#!/bin/bash -l
# ==============================================================================
# SLURM job: Pool-512 evaluation for mhcSFM
# Runs eval_pool512_fast on all folds for identity, 80%, 60% OOD
# No GPU needed — just memory for loading cosine_sim matrices (~4.3 GB each)
#
# Usage:  sbatch scripts/euler_pool512_mhcsfm.slurm
# ==============================================================================

#SBATCH --job-name=mhcsfm-p512
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mhcsfm_pool512_%j.out
#SBATCH --error=logs/mhcsfm_pool512_%j.err

# --- Environment setup ---
module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate

export PYTHONUNBUFFERED=1

CALM_DIR="${HOME}/CALM-0.1.0"
OUTPUT_BASE="/cluster/scratch/${USER}/mhcsfm/output/mhcsfm_full"

cd ${CALM_DIR}
mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "mhcSFM Pool-512 Evaluation"
echo "  Start time: $(date)"
echo "============================================"

for SPLIT in identity_100 mmseqs_080 mmseqs_060; do
    echo ""
    echo "========================================"
    echo "=== ${SPLIT} ==="
    echo "========================================"
    for FOLD in 0 1 2 3 4; do
        EVAL_DIR="${OUTPUT_BASE}/mhcsfm-${SPLIT}-fold${FOLD}/eval/fold_${FOLD}"
        if [ -f "${EVAL_DIR}/cosine_sim.pt" ]; then
            echo ""
            echo "--- ${SPLIT} fold ${FOLD} ---"
            python -m calm.encoder.eval_pool512_fast \
                --eval_dir "${EVAL_DIR}" \
                --pool_size 512 \
                --n_trials 100 \
                --max_queries 200
        else
            echo "  Skipping ${SPLIT} fold ${FOLD} (no cosine_sim.pt)"
        fi
    done
done

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "============================================"
