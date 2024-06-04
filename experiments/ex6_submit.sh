#!/bin/bash

#SBATCH --job-name=experiment6        # job name
#SBATCH --output=slurm_out/ex6_%A_%a.out    # use %A for job id, %a for task id
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=02:00:00                   # wall-clock time limit

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12                #76

#SBATCH --array=0-14%4     # use job arrays to do multiple tasks indexed by this array, max 4 jobs simultaneously

# Set up modules.
module purge
module load compiler/gnu/11
module load devel/cuda/11.8
module load lib/hdf5/1.12

# training on 5 different seeds to get a more stable result

source ../../venv/ensemblefusion/bin/activate      # Activate your virtual environment.

seeds=("0" "1" "2" "3" "4")
methods=("p_m_ft" "m_p_ft" "m_ft_p_ft")

num_seeds=${#seeds[@]}
num_methods=${#methods[@]}


seed_index=$((SLURM_ARRAY_TASK_ID % num_seeds))
method_index=$(( (SLURM_ARRAY_TASK_ID / num_seeds) % num_methods ))

seed=${seeds[seed_index]}
method=${methods[method_index]}


python -u experiments/ex6_order.py "$method" "$seed"
