#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=my_python_job
#SBATCH --time=24:00:00
#SBATCH --output=job_output.txt

cd $SLURM_SUBMIT_DIR

module load python/3.8.6
module load openmpi
module load gcc/7.2.0

pip install --user numpy pandas tsfresh astropy mpi4py

srun -n 16 python3 Preproces.py
