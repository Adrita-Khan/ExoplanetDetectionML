#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=15:00:00
#SBATCH --output=job_output.txt
#SBATCH --error=job_errors.txt

cd $SLURM_SUBMIT_DIR

module load python/3.8.6
module load openmpi
module load gcc/7.2.0

pip install --user numpy pandas tsfresh astropy mpi4py statsmodels scikit-learn-intelex optuna

mpirun -np 16 python3 Random_Forest_Hyper.py
