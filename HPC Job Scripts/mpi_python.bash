#!/bin/bash

# Set the number of nodes and processes per node
#PBS -l nodes=2:ppn=8

# Set the maximum walltime
#PBS -l walltime=00:30:00

# Set the output file name
#PBS -o output.txt

# Set the queue (replace <queue-name> with the actual queue name)
#PBS -q <queue-name>

# Load the MPI module
module load openmpi

# Run the MPI program
mpirun -np 16 python my_mpi_program.py
