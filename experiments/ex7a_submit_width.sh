#!/bin/bash

#SBATCH --job-name=experiment7a        # job name
#SBATCH --output=slurm_out/ex7a_comparison_%A_%a.out    # use %A for job id, %a for task id
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=02:00:00                   # wall-clock time limit

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12                #76

                             # --aray=0-29%4 for first few models (first commented out)
#SBATCH --array=0-7%4     # use job arrays to do multiple tasks indexed by this array, max 4 jobs simultaneously

# Set up modules.
module purge
module load compiler/gnu/11
module load devel/cuda/11.8
module load lib/hdf5/1.12

# training on 5 different seeds to get a more stable result

source ../../venv/ensemblefusion/bin/activate      # Activate your virtual environment.

widths=("16" "32" "64" "128" "256" "512" "1024" "2048")
num_widths=${#widths[@]}
width_index=$((SLURM_ARRAY_TASK_ID % num_widths))
width=${widths[width_index]}

python -u experiments/ex7_train.py "$width" 4 2 0
python -u experiments/ex7_comparison.py "nt" "$width" 4 2 0
python -u experiments/ex7_comparison.py "full_ensemble" "$width" 4 2 0
python -u experiments/ex7_comparison.py "argmax" "$width" 4 2 0
python -u experiments/ex7_plotter.py "$width" 4 2 0