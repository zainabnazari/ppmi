#!/bin/bash
#SBATCH --job-name=e1_ir2
#SBATCH --partition=regular1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:50:00

cd ${SLURM_SUBMIT_DIR}

conda init bash
source /home/znazari/.bashrc

conda activate Zainab-env


python3 quant_easy_v06.py



