#!/bin/bash -l
# ==============================================================================
# SLURM job: Preprocess JASPAR data for tSFM
#
# This job downloads JASPAR profiles, fetches protein sequences from UniProt,
# and embeds everything with ESM-2 + DNABERT-2.
#
# Usage:  sbatch scripts/euler_preprocess_jaspar.slurm
# ==============================================================================

#SBATCH --job-name=tsfm-preprocess
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# --- Configuration ---
# Adjust these paths to match your Euler setup
CALM_DIR="${HOME}/CALM-0.1.0"           # Where you copied the CALM code
OUTPUT_DIR="${CALM_DIR}/data/jaspar"     # Where preprocessed data goes
MAX_PROFILES=0                           # 0 = all profiles (~2059)
SAMPLES_PER_PWM=10                       # DNA sequences sampled per PWM (10 = ~20K pairs)

# --- Environment setup ---
module load stack/2024-06 gcc/12.2.0 python/3.11.6
source ~/calm_env/bin/activate

# Create log directory
mkdir -p ${CALM_DIR}/logs

echo "============================================"
echo "tSFM Preprocessing — JASPAR"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Node:          $(hostname)"
echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CALM_DIR:      ${CALM_DIR}"
echo "  OUTPUT_DIR:    ${OUTPUT_DIR}"
echo "  MAX_PROFILES:  ${MAX_PROFILES}"
echo "  SAMPLES/PWM:   ${SAMPLES_PER_PWM}"
echo "  Start time:    $(date)"
echo "============================================"

cd ${CALM_DIR}

srun python -m calm.preprocess.jaspar \
    --output_dir ${OUTPUT_DIR} \
    --max_profiles ${MAX_PROFILES} \
    --device cuda \
    --esm_batch_size 2 \
    --dna_batch_size 8 \
    --samples_per_pwm ${SAMPLES_PER_PWM}

echo ""
echo "============================================"
echo "  End time: $(date)"
echo "  Output:   ${OUTPUT_DIR}"
echo "============================================"
