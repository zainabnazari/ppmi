#!/bin/bash
#SBATCH --job-name=blq
#SBATCH --partition=regular1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

cd ${SLURM_SUBMIT_DIR}

conda init bash
source /home/znazari/.bashrc

conda activate Zainab-env


python3 quant_easy_bl.py



