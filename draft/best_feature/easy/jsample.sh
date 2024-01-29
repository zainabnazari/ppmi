#!/bin/bash
#SBATCH --job-name=my_dask_job
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --partition=regular2

module purge
module load gnu8/8.3.0
module load openmpi3/3.1.4
module --ignore-cache load "python/3.8.5"
source /home/znazari/.bashrc

# Activate your conda environment
conda activate Zainab-env  # Change this with your actual environment name


# Change to the directory where the SLURM job was submitted
cd $SLURM_SUBMIT_DIR

# Run your Python script using mpirun
mpirun -n 3 python sample.py > time.txt

