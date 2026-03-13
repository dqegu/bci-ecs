#!/bin/bash -l
#SBATCH -p l4,scavenger_l4,ecsstudents_l4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@soton.ac.uk
#SBATCH -o logs/ssvep_%j.out
#SBATCH -e logs/ssvep_%j.err


module load conda/python3
source ~/.bashrc
conda activate my-ssvep-env

cd ~/bci-ecs

export WANG_DATASET_DIR="$PWD/data"
export SSVEP_OUTPUT_DIR="$PWD/results"

mkdir -p logs
mkdir -p results

python ssvep_cca_pipeline.py
