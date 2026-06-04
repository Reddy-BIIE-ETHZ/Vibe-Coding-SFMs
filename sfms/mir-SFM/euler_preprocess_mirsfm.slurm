#!/bin/bash -l
# ==============================================================================
# SLURM job: Preprocess miR-SFM dataset (embed miRNA + target with DNABERT-2)
#
# Expected runtime: ~45 min (139 miRNAs fast, ~200K targets @ batch_size=16)
# Output: /cluster/scratch/$USER/mirsfm/data/
# ==============================================================================

#SBATCH --job-name=mirsfm-prep
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/mirsfm_preprocess_%j.out
#SBATCH --error=logs/mirsfm_preprocess_%j.err

CALM_DIR="${HOME}/CALM-0.1.0"
DATA_FILE="${CALM_DIR}/data/mirsfm/mirsfm_training.tsv"
OUTPUT_DIR="/cluster/scratch/${USER}/mirsfm/data"

module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate
export PYTHONUNBUFFERED=1
export TRITON_DISABLE=1
export DNABERT2_DISABLE_FLASH=1

mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "miR-SFM Preprocessing"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  DATA_FILE:   ${DATA_FILE}"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  Start time:  $(date)"
echo "============================================"

cd ${CALM_DIR}

srun python -m calm.preprocess.mirsfm \
    --training_file ${DATA_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --device cuda \
    --dna_batch_size 16

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "============================================"
