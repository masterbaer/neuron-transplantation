'''
This python file compares NT to output averaging and selecting the best ensemble member with different
widths, heights and ensemble sizes.
'''

import sys
from pathlib import Path

from torch import nn
from torch.optim.lr_scheduler import ConstantLR

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fusion_methods.classical_ensembles import evaluate_accuracy_output_averaging
from fusion_methods.neuron_transplantation import fuse_ensemble, fuse_ensemble_iterative, fuse_ensemble_hierarchical
from model import AdaptiveNeuralNetwork2
import torch
from dataloader import get_dataloader_from_name
from train import train_model
from train_helper import set_all_seeds, evaluate_model

if __name__ == "__main__":

    dataset_name = "svhn"
    method = sys.argv[1]  # "argmax", "full_ensemble", "nt"
    model_width = int(sys.argv[2])  # standard 512 (32,64,128,256,512,1024, 2048)
    model_depth = int(sys.argv[3])  # standard 4  (1, 2, 4, 8, 16)
    num_models = int(sys.argv[4])  # standard 2 , 4, 8, 16
    master_seed = int(sys.argv[5])  # one seed is enough as it is not an important comparison but a wage experiment

    print("method: ", method, "model_width: ", model_width, "model_depth: ", model_depth, "num_models: ", num_models)
    print("master seed: ", master_seed)

    b = 256
    e = 30
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
            model_dict = torch.load(f'models/ex7_{model_width}_{model_depth}_{master_seed}_{i}.pt', map_location=device)
            input_dim = image_shape[1] * image_shape[2] * image_shape[3]
            model = AdaptiveNeuralNetwork2(input_dim=input_dim, output_dim=num_classes,
                                           layer_width=model_width, num_layers=model_depth).to(device)
            model.load_state_dict(model_dict)
            models.append(model)

    except FileNotFoundError:
        print("models not found")
        exit()

    # fuse and finetune , print all accuracies and report the best accuracy reached ,
    # as well as the ensemble_acc and best_individual_acc

    if method == "argmax":

        best_individual_acc = 0.0
        for i, model in enumerate(models):
            set_all_seeds(master_seed)
            train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, batch_size=b,
                                                                               root=dataset_name)
            model.eval()
            _, test_accuracy = evaluate_model(model, test_loader, device)
            if test_accuracy > best_individual_acc:
                best_individual_acc = test_accuracy
            print(f"model {i} test accuracy: {test_accuracy}")
            torch.save(best_individual_acc, f'out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_best_individual_acc.pt')

    if method == "full_ensemble":
        set_all_seeds(master_seed)
        train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, batch_size=b,
                                                                           root=dataset_name)
        acc_ensemble = evaluate_accuracy_output_averaging(test_loader, models, device)
        print("full ensemble acc: ", acc_ensemble)
        torch.save(acc_ensemble, f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_full_ensemble_acc.pt")

    if method == "nt":

        for model in models:
            model.to("cpu")
        fused_model = fuse_ensemble(models, example_inputs)

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

        torch.save(test_acc_list, f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_acc.pt")

    if method == "nt_iterative":
        for model in models:
            model.to("cpu")
        fused_model = fuse_ensemble_iterative(models, example_inputs)

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

        torch.save(test_acc_list, f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_iterative_acc.pt")
    ## this experiment tries to find out how to merge multiple models efficiently:

    # a) make a giant model by concatenating all ensemble members and jointly prune them
    # b) merge 2 models, finetune them for 3 epochs, and then feed in another model...

    if method == "nt_hierarchical":
        for model in models:
            model.to("cpu")
        fused_model = fuse_ensemble_hierarchical(models, example_inputs)

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

        torch.save(test_acc_list, f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_hierarchical_acc.pt")