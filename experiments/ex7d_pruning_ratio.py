'''
Sparsities: [0, 25, 50, 75, 87.5, 93.75, 95]
We check the fusion with different sparsities for the pruning after concatenation.
The sparsity of 1-1/k (for k models) leads to the original architecture.
'''


import sys
from pathlib import Path

from torch import nn
from torch.optim.lr_scheduler import ConstantLR
import torch_pruning as tp

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fusion_methods.classical_ensembles import evaluate_accuracy_output_averaging
from fusion_methods.neuron_transplantation import fuse_ensemble, fuse_ensemble_iterative, fuse_ensemble_hierarchical, \
    concat_models, getLastLayerName, get_module_by_name
from model import AdaptiveNeuralNetwork2
import torch
from dataloader import get_dataloader_from_name
from train import train_model
from train_helper import set_all_seeds, evaluate_model

if __name__ == "__main__":

    dataset_name = "svhn"
    method = sys.argv[1]  # "argmax", "full_ensemble", "nt"
    model_width = 512  # standard 512 (32,64,128,256,512,1024, 2048)
    model_depth = 4  # standard 4  (1, 2, 4, 8, 16)
    num_models = int(sys.argv[2])  # standard 2 , 4, 8, 16
    sparsity = float(sys.argv[3])  # sparsities: [0, 25, 50, 75, 87.5, 93.75, 95]
    master_seed = 0  # one seed is enough as it is not an important comparison but a wage experiment

    print("method: ", method, "sparsity: ", sparsity, "num_models: ", num_models)
    print("master seed: ", master_seed)

    b = 256
    e = 20
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
            torch.save(best_individual_acc, f'out/ex7d_{model_width}_{model_depth}_{num_models}_{master_seed}_best_individual_acc.pt')

    if method == "full_ensemble":
        set_all_seeds(master_seed)
        train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, batch_size=b,
                                                                           root=dataset_name)
        acc_ensemble = evaluate_accuracy_output_averaging(test_loader, models, device)
        print("full ensemble acc: ", acc_ensemble)
        torch.save(acc_ensemble, f"out/ex7d_{model_width}_{model_depth}_{num_models}_{master_seed}_full_ensemble_acc.pt")

    if method == "nt":

        for model in models:
            model.to("cpu")

        ###  start fusing

        # concatenation
        fused_model = concat_models(models)

        last_layer_name = getLastLayerName(fused_model)
        last_layer = get_module_by_name(fused_model, last_layer_name)

        # sparsity chosen in command line arguments is fed into the pruner
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner = tp.pruner.MagnitudePruner(
            fused_model,
            example_inputs,
            importance=imp,
            pruning_ratio=sparsity,
            root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
            ignored_layers=[last_layer],
        )
        pruner.step()

        print("printing modules: ")
        for module in fused_model.modules():
            print(module)

        ### fusing done

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

        torch.save(test_acc_list, f"out/ex7d_{num_models}_{sparsity}_{master_seed}_nt_acc.pt")
