#!/bin/bash
#SBATCH --job-name=kpc-dask-2node
#SBATCH --partition=long2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00


module purge
module load gnu11 


source /home/znazari/.bashrc

conda activate Zainab-env #you have to change this with your environment

cd $SLURM_SUBMIT_DIR

python main_joblib_algorithm.py 

