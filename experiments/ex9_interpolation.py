'''
The main property of Neuron Transplantation. Here we fuse with different transplantation percentages.
This is comparable to linear interpolation between two models, just with transplantation.
We can observe two things:
a) The post-fusion loss increases for 50% transplantation.
b) The fine-tune accuracy increases for 50% transplantation.
The newly transplanted neurons make the model worse at first while the performance gets better after fine-tuning.
'''

import sys
from pathlib import Path

from torch import nn
from torch.optim.lr_scheduler import ConstantLR

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fusion_methods.classical_ensembles import evaluate_accuracy_output_averaging
from fusion_methods.neuron_transplantation import fuse_ensemble, fuse_ensemble_iterative, fuse_ensemble_hierarchical, \
    getLastLayerName, get_module_by_name, concat_models
from model import AdaptiveNeuralNetwork2
import torch
from dataloader import get_dataloader_from_name
from train import train_model
from train_helper import set_all_seeds, evaluate_model
import torch_pruning as tp

if __name__ == "__main__":

    dataset_name = "svhn"
    method = sys.argv[1]  # "full_ensemble", "nt"
    transfer_rate = float(sys.argv[2])  # standard 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
    master_seed = 0
    num_models = 2

    print("transfer rate: ", transfer_rate)

    b = 256
    e = 3
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print("loading models")

    set_all_seeds(master_seed)
    train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, root=dataset_name,
                                                                       batch_size=b)
    image_shape = None
    example_inputs = None
    for images, labels in train_loader:
        image_shape = images.shape
        example_inputs = images
        break
    example_inputs.to("cpu")

    input_dim = image_shape[1] * image_shape[2] * image_shape[3]

    models = []

    try:
        for i in range(num_models):
            model_dict = torch.load(f'models/ex7_512_4_{master_seed}_{i}.pt', map_location=device)
            input_dim = image_shape[1] * image_shape[2] * image_shape[3]
            model = AdaptiveNeuralNetwork2(input_dim=input_dim, output_dim=num_classes,
                                           layer_width=512, num_layers=4).to(device)
            model.load_state_dict(model_dict)
            models.append(model)

    except FileNotFoundError:
        print("models not found")
        exit()

    # fuse and finetune , print all accuracies and report the best accuracy reached ,
    # as well as the ensemble_acc and best_individual_acc

    if method == "full_ensemble":
        set_all_seeds(master_seed)
        train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, batch_size=b,
                                                                           root=dataset_name)
        acc_ensemble = evaluate_accuracy_output_averaging(test_loader, models, device)
        print("full ensemble acc: ", acc_ensemble)
        torch.save(acc_ensemble, f"out/ex9_full_ensemble_acc.pt")

    if method == "nt":

        for model in models:
            model.to("cpu")

        #fused_model = fuse_ensemble(models, example_inputs)
        fused_model = None

        if transfer_rate == 0.0:
            fused_model = models[0]
            print("model A")
        elif transfer_rate == 1.0:
            fused_model = models[1]
            print("model B")
        else:
            # prune A and B according to sparsity and then combine

            last_layer_name_A = getLastLayerName(models[0])
            last_layer_A = get_module_by_name(models[0], last_layer_name_A)
            imp_A = tp.importance.MagnitudeImportance(p=2)
            pruner_A = tp.pruner.MagnitudePruner(
                models[0],
                example_inputs,
                importance=imp_A,
                pruning_ratio=transfer_rate,
                root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
                ignored_layers=[last_layer_A],
            )
            pruner_A.step()

            last_layer_name_B = getLastLayerName(models[1])
            last_layer_B = get_module_by_name(models[1], last_layer_name_B)
            imp_B = tp.importance.MagnitudeImportance(p=2)
            pruner_B = tp.pruner.MagnitudePruner(
                models[1],
                example_inputs,
                importance=imp_B,
                pruning_ratio=1-transfer_rate,
                root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
                ignored_layers=[last_layer_B],
            )
            pruner_B.step()

            fused_model = concat_models(models)

        # now do the finetuning

        set_all_seeds(master_seed)
        train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, batch_size=b,
                                                                           root=dataset_name)
        fused_model.to(device)

        loss, accuracy = evaluate_model(fused_model, test_loader, device)
        print(accuracy)
        test_acc_list = [accuracy]

        optimizer = torch.optim.SGD(fused_model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)

        test_acc_list_finetune = train_model(model=fused_model, optimizer=optimizer, criterion=criterion,
                                             scheduler=scheduler,
                                             train_loader=train_loader, valid_loader=test_loader, e=e,
                                             device=device)
        test_acc_list = test_acc_list + test_acc_list_finetune

        torch.save(test_acc_list, f"out/ex9_{transfer_rate}_nt_acc.pt")
