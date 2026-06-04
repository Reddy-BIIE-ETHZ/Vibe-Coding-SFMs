#!/bin/bash -l
#SBATCH --job-name=esfm-eval-filt
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/esfm_eval_filtered_%j.out
#SBATCH --error=logs/esfm_eval_filtered_%j.err

CALM_DIR="${HOME}/CALM-0.1.0"
DATA_DIR="/cluster/scratch/${USER}/esfm"
OUTPUT_DIR="/cluster/scratch/${USER}/esfm/output/esfm_full"
FILTER_DIR="${CALM_DIR}/audit/leakage_verification/esfm/filters"

module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate

export PYTHONPATH=src:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "============================================"
echo "eSFM Filtered Eval (strict-OOD)"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Start time:  $(date)"
echo "============================================"

cd ${CALM_DIR}

python scripts/eval_esfm_pool512_filtered.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --filter_dir "${FILTER_DIR}" \
    --pool_size 512 \
    --n_trials 100 \
    --seed 42 \
    --splits mmseqs_040 mmseqs_060 mmseqs_080

echo "============================================"
echo "  End time: $(date)"
echo "============================================"
