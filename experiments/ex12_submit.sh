#!/bin/bash

#SBATCH --job-name=experiment12        # job name
#SBATCH --output=slurm_out/ex12_%A_%a.out    # use %A for job id, %a for task id
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=00:30:00                   # wall-clock time limit

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12                #76

                             # --aray=0-29%4 for first few models (first commented out)
#SBATCH --array=0-29%4     # use job arrays to do multiple tasks indexed by this array, max 4 jobs simultaneously

# Set up modules.
module purge
module load compiler/gnu/11
module load devel/cuda/11.8
module load lib/hdf5/1.12

# training on 5 different seeds to get a more stable result

source ../../venv/ensemblefusion/bin/activate      # Activate your virtual environment.

#seeds=("0" "1" "2" "3" "4")
#datasets=("cifar10" "mnist" "svhn")
#models=("smallnn" "lenet")   # first few models

methods=("avg" "nt" "ot")
widths=("256" "384" "512" "768" "1024" "1536" "2048" "4096" "8192" "16384")      # vgg11 on cifar10 and cifar100

num_methods=${#methods[@]}
num_widths=${#widths[@]}


method_index=$((SLURM_ARRAY_TASK_ID % num_methods))
width_index=$(( (SLURM_ARRAY_TASK_ID / num_methods) % num_widths ))

method=${methods[method_index]}
width=${widths[width_index]}


python -u experiments/ex12_comparison_time_memory.py "$method" "$width"
