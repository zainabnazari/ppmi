#!/bin/bash
#SBATCH --job-name=dask_job
#SBATCH --partition=long1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=30:00:00

# Load necessary modules (if required)
# module load your_module
source /home/znazari/.bashrc

# Activate your Python environment
conda activate Zainab-env
cd $SLURM_SUBMIT_DIR

# Run your Dask code
python parallel_best_proteomic.py > parallel_result.txt

