#!/bin/bash

#SBATCH --job-name=experiment7e        # job name
#SBATCH --output=slurm_out/ex7e_comparison_%A_%a.out    # use %A for job id, %a for task id
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=02:00:00                   # wall-clock time limit

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12                #76

#SBATCH --array=0-1%4     # use job arrays to do multiple tasks indexed by this array, max 4 jobs simultaneously

# Set up modules.
module purge
module load compiler/gnu/11
module load devel/cuda/11.8
module load lib/hdf5/1.12

# training on 5 different seeds to get a more stable result

source ../../venv/ensemblefusion/bin/activate      # Activate your virtual environment.

widths=("165" "512")
#small_model = AdaptiveNeuralNetwork2(input_dim=input_dim, output_dim=num_classes, layer_width=165, num_layers=4)
#large_model = AdaptiveNeuralNetwork2(input_dim=input_dim, output_dim=num_classes, layer_width=512, num_layers=4)
#print("4 * small model parameters: ", 4 * sum(p.numel() for p in small_model.parameters()))
# 2360820
#print("large model parameters: ", sum(p.numel() for p in large_model.parameters()))
# 2364416

num_widths=${#widths[@]}
width_index=$((SLURM_ARRAY_TASK_ID % num_widths))
width=${widths[width_index]}

python -u experiments/ex7_train.py "$width" 4 4 0
python -u experiments/ex7_comparison.py "nt" "$width" 4 4 0
python -u experiments/ex7_comparison.py "full_ensemble" "$width" 4 4 0
python -u experiments/ex7_comparison.py "argmax" "$width" 4 4 0

# small model ensemble: 84.74
# nt of larger model after 30epochs: 84.96 while the first 29 epochs were slightly less than 84.74
# --> in terms of capacity in this one example nt and a smaller (untouched) ensemble are similar