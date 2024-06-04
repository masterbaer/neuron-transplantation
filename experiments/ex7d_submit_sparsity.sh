#!/bin/bash

#SBATCH --job-name=experiment7d        # job name
#SBATCH --output=slurm_out/ex7d_sparsity_%A_%a.out    # use %A for job id, %a for task id
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=05:00:00                   # wall-clock time limit

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12                #76

#SBATCH --array=0-31%4     # use job arrays to do multiple tasks indexed by this array, max 4 jobs simultaneously

# Set up modules.
module purge
module load compiler/gnu/11
module load devel/cuda/11.8
module load lib/hdf5/1.12

# training on 5 different seeds to get a more stable result

source ../../venv/ensemblefusion/bin/activate      # Activate your virtual environment.

sparsities=("0.0" "0.25" "0.5" "0.75" "0.875" "0.9375" "0.95" "0.99")
counts=("16" "8" "4" "2")

num_sparsities=${#sparsities[@]}
num_counts=${#counts[@]}


sparsity_index=$((SLURM_ARRAY_TASK_ID % num_sparsities))
count_index=$(( (SLURM_ARRAY_TASK_ID / num_sparsities) % num_counts))

sparsity=${sparsities[sparsity_index]}
count=${counts[count_index]}

python -u experiments/ex7d_pruning_ratio.py "full_ensemble" "$count" "$sparsity"
python -u experiments/ex7d_pruning_ratio.py "argmax" "$count" "$sparsity"
python -u experiments/ex7d_pruning_ratio.py "nt" "$count" "$sparsity"