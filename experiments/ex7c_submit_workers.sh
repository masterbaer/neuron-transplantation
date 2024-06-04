#!/bin/bash

#SBATCH --job-name=experiment7c        # job name
#SBATCH --output=slurm_out/ex7c_comparison_%A_%a.out    # use %A for job id, %a for task id
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=05:00:00                   # wall-clock time limit

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12                #76

#SBATCH --array=0-19%4     # use job arrays to do multiple tasks indexed by this array, max 4 jobs simultaneously

# Set up modules.
module purge
module load compiler/gnu/11
module load devel/cuda/11.8
module load lib/hdf5/1.12

# training on 5 different seeds to get a more stable result

source ../../venv/ensemblefusion/bin/activate      # Activate your virtual environment.

seeds=("0" "1" "2" "3" "4")
counts=("16" "8" "4" "2")

num_seeds=${#seeds[@]}
num_counts=${#counts[@]}


seed_index=$((SLURM_ARRAY_TASK_ID % num_seeds))
count_index=$(( (SLURM_ARRAY_TASK_ID / num_seeds) % num_counts))

seed=${seeds[seed_index]}
count=${counts[count_index]}


python -u experiments/ex7_train.py 512 4 "$count" "$seed"
python -u experiments/ex7_comparison.py "nt" 512 4 "$count" "$seed"
python -u experiments/ex7_comparison.py "nt_iterative" 512 4 "$count" "$seed"
python -u experiments/ex7_comparison.py "nt_hierarchical" 512 4 "$count" "$seed"
python -u experiments/ex7_comparison.py "full_ensemble" 512 4 "$count" "$seed"
python -u experiments/ex7_comparison.py "argmax" 512 4 "$count" "$seed"
python -u experiments/ex7_plotter.py 512 4 "$count" "$seed"