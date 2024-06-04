#!/bin/bash

#SBATCH --job-name=experiment7b        # job name
#SBATCH --output=slurm_out/ex7b_comparison_%A_%a.out    # use %A for job id, %a for task id
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=02:00:00                   # wall-clock time limit

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12                #76

                             # --aray=0-29%4 for first few models (first commented out)
#SBATCH --array=0-5%4     # use job arrays to do multiple tasks indexed by this array, max 4 jobs simultaneously

# Set up modules.
module purge
module load compiler/gnu/11
module load devel/cuda/11.8
module load lib/hdf5/1.12

# training on 5 different seeds to get a more stable result

source ../../venv/ensemblefusion/bin/activate      # Activate your virtual environment.

depths=("1" "3" "5" "7" "9" "11")
num_depths=${#depths[@]}
depth_index=$((SLURM_ARRAY_TASK_ID % num_depths))
depth=${depths[depth_index]}

python -u experiments/ex7_train.py 512 "$depth" 2 0
python -u experiments/ex7_comparison.py "nt" 512 "$depth" 2 0
python -u experiments/ex7_comparison.py "full_ensemble" 512 "$depth" 2 0
python -u experiments/ex7_comparison.py "argmax" 512 "$depth" 2 0
python -u experiments/ex7_plotter.py 512 "$depth" 2 0