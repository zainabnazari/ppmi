#!/bin/bash
#SBATCH --job-name=v06n
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

conda list

python3 main_v06.py



