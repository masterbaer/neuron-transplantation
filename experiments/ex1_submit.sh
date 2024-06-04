#!/bin/bash
# Batch file to run experiment 1. Use sbatch ex1_submit.sh .
# Some relevant modules and the virtual environment is loaded before running the python program.

#SBATCH --job-name=pruning_experiment1        # job name
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=01:00:00                   # wall-clock time limit
#SBATCH --output=slurm_out/ex1.out

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

# see https://gitlab.jsc.fz-juelich.de/MLDL_FZJ/juhaicu/jsc_public/sharedspace/teaching/haicore-tutorial/-/blob/main/03SubmitExample/exercise1.sbatch?ref_type=heads
# for a guide on slurm

source ../../venv/ensemblefusion/bin/activate      # Activate your virtual environment.

mpirun python -u experiments/ex1_parallel_train.py
# map by works with core/socket/node, see https://www.nhr.kit.edu/userdocs/haicore/batch_slurm_mpi/