#!/bin/bash
#SBATCH --job-name=testconda
#SBATCH --partition=regular2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00

cd ${SLURM_SUBMIT_DIR}

conda init bash
source /home/znazari/.bashrc

conda activate Zainab-env
python file.py

echo "the job succesfully finished! hooray!"
