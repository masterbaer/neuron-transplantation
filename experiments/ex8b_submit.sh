#!/bin/bash

#SBATCH --job-name=ex8b        # job name
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=02:00:00                   # wall-clock time limit
#SBATCH --output=slurm_out/ex8b.out

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12                #76

# Set up modules.
module purge
module load compiler/gnu/11
module load devel/cuda/11.8
module load mpi/openmpi/4.0
module load lib/hdf5/1.12

source ../../venv/ensemblefusion/bin/activate      # Activate your virtual environment.

mpirun python -u experiments/ex8b_synchronous_sgd_with_nt.py "avg"
mpirun python -u experiments/ex8b_synchronous_sgd_with_nt.py "nt"