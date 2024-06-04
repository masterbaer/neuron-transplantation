#!/bin/bash

#SBATCH --job-name=experiment11        # job name
#SBATCH --output=slurm_out/ex11_comparison_%A_%a.out    # use %A for job id, %a for task id
#SBATCH --partition=normal            # queue for resource allocation
#SBATCH --time=08:00:00                   # wall-clock time limit

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12                #76

                             # --aray=0-29%4 for first few models (first commented out)
#SBATCH --array=0-9%4     # use job arrays to do multiple tasks indexed by this array, max 4 jobs simultaneously

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

seeds=("0" "1" "2" "3" "4") #
datasets=("cifar10" "cifar100") #
#models=("vgg11")      # vgg11 on cifar10 and cifar100
models=("resnet18")
# models=("vit")


num_seeds=${#seeds[@]}
num_datasets=${#datasets[@]}
num_models=${#models[@]}

seed_index=$((SLURM_ARRAY_TASK_ID % num_seeds))
dataset_index=$(( (SLURM_ARRAY_TASK_ID / num_seeds) % num_datasets ))
model_index=$((SLURM_ARRAY_TASK_ID / (num_seeds * num_datasets)))

model=${models[model_index]}
dataset=${datasets[dataset_index]}
seed=${seeds[seed_index]}

python -u experiments/ex11_train.py "$dataset" "$model" "$seed"
python -u experiments/ex11_comparison.py "$dataset" "$model" "$seed" "argmax"
python -u experiments/ex11_comparison.py "$dataset" "$model" "$seed" "full_ensemble"
python -u experiments/ex11_comparison.py "$dataset" "$model" "$seed" "avg_ft"
python -u experiments/ex11_comparison.py "$dataset" "$model" "$seed" "nt_ft"
python -u experiments/ex11_comparison.py "$dataset" "$model" "$seed" "ot_ft"
python -u experiments/ex11_comparison.py "$dataset" "$model" "$seed" "model0_distill"
python -u experiments/ex11_comparison.py "$dataset" "$model" "$seed" "avg_distill"
python -u experiments/ex11_comparison.py "$dataset" "$model" "$seed" "nt_distill"
python -u experiments/ex11_comparison.py "$dataset" "$model" "$seed" "ot_distill"
python -u experiments/ex11_plotter.py "$dataset" "$model" "$seed"
#python -u experiments/ex11_delete_models.py "$dataset" "$model" "$seed"
