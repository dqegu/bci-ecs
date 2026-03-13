#!/bin/bash -l
#SBATCH -J ssvep_cca
#SBATCH -p serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@soton.ac.uk
#SBATCH -o logs/ssvep_%j.out
#SBATCH -e logs/ssvep_%j.err

set -euo pipefail

mkdir -p logs

module load conda/python3
conda activate my-ssvep-env

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"

# ---- paths ----
PROJECT_DIR="$PWD"
DATA_ROOT="$PWD/data"
RESULTS_DIR="$PWD/results"
ZIP_PATH="$DATA_ROOT/wang_dataset.zip"

mkdir -p "$DATA_ROOT"
mkdir -p "$RESULTS_DIR"

# ---- dataset download (only if not already present) ----
# Replace this folder name if your zip extracts into a different name.
EXTRACTED_DATASET_DIR="$DATA_ROOT/wang_dataset"

if [ ! -d "$EXTRACTED_DATASET_DIR" ]; then
    echo "Dataset folder not found, preparing dataset..."

    if [ ! -f "$ZIP_PATH" ]; then
        echo "Downloading dataset from Google Drive..."
        python -m gdown "https://drive.google.com/uc?id=1xGu1LDg_dOutafB0VI7VzHKW7Y0ohD34" -O "$ZIP_PATH"
    fi

    echo "Unzipping dataset..."
    unzip -q "$ZIP_PATH" -d "$DATA_ROOT"
fi

# If unzip creates a different folder name, set this correctly.
export WANG_DATASET_DIR="$EXTRACTED_DATASET_DIR"
export SSVEP_OUTPUT_DIR="$RESULTS_DIR"

echo "WANG_DATASET_DIR=$WANG_DATASET_DIR"
echo "SSVEP_OUTPUT_DIR=$SSVEP_OUTPUT_DIR"

python ssvep_cca_pipeline.py

echo "Job finished."