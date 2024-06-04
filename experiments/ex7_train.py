'''
Experiment 7 tries Neuron Transplantation with different settings.
Ex7_train trains num_models of depth model_depth and width model_width for a given random seed.
Ex7_comparison compares NT to output-averaging and selecting the best ensemble member.
The submit files use these python files to try NT on
a) different widths
b) different depths
c) different ensemble size (with different reduction schemes: iterative, recursive, joint).
d) different sparsities (only for the sparsity 1-1/k NT is a fusion method).

E.g. use sbatch ex7a_submit_width.sh or run the programs manually.
'''

# TODO: test the method for many different ANN architecture types: compare width with fixed depth
# and compare: a) how much improvement compared to model 0
#              b) how many finetuning epochs needed, is there any improvement?

# TODO look into https://arxiv.org/pdf/1910.05653.pdf S10

import sys
from pathlib import Path

from torch import nn
from torch.optim.lr_scheduler import ConstantLR

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import AdaptiveNeuralNetwork2
import torch
from dataloader import get_dataloader_from_name
from train import train_model
from train_helper import set_all_seeds

if __name__ == "__main__":

    # command line arguments: dataset, model(dataset), seed

    # argv[0] is the file name ex11_comparison.py
    dataset_name = "svhn"  # chose svhn as ex11 had best results there, so we might see noticable effects here
    model_width = int(sys.argv[1])  # standard 512 (32,64,128,256,512,1024, 2048)
    model_depth = int(sys.argv[2])  # standard 4  (1, 2, 4, 8, 16)
    num_models = int(sys.argv[3])  # standard 2 , 4, 8, 16

    master_seed = int(sys.argv[4])  # one seed is enough as it is not an important comparison but a wage experiment

    print("model_width: ", model_width, "model_depth: ", model_depth, "num_models: ", num_models)
    print("master seed: ", master_seed)

    b = 256
    e = 100
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print("training models")

    for i in range(num_models):

        try:
            model_dict = torch.load(f'models/ex7_{model_width}_{model_depth}_{master_seed}_{i}.pt',
                                    map_location=device)
            print("trained model already exists")
            continue
        except FileNotFoundError:
            print(f"training model {model_width}_{model_depth}_{master_seed}_{i}")

        set_all_seeds(100 * master_seed + i)  # local seeds: (0,1,2,3) or (100,101,102,103) if master seed is 0 or 1

        train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, root=dataset_name,
                                                                           batch_size=b)
        image_shape = None
        for images, labels in train_loader:
            image_shape = images.shape
            example_inputs = images
            break

        input_dim = image_shape[1] * image_shape[2] * image_shape[3]

        model = AdaptiveNeuralNetwork2(input_dim=input_dim, output_dim=num_classes,
                                       layer_width=model_width, num_layers=model_depth).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)

        # train model
        val_acc_list = train_model(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                               train_loader=train_loader, valid_loader=valid_loader, e=e, device=device)

        torch.save(model.state_dict(), f'models/ex7_{model_width}_{model_depth}_{master_seed}_{i}.pt')
