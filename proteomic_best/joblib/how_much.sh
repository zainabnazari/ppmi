#!/bin/bash
#SBATCH --job-name=joblib
#SBATCH --partition=long1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1		
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00


module purge


source /home/znazari/.bashrc

conda activate Zainab-env # my environment

cd $SLURM_SUBMIT_DIR

python  how_much_machine.py > ttt.txt

