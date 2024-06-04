'''
This experiment checks in what order we should do the operations of NT:
a)/b) Prune ensemble members, (fine-tune for b), ) concatenate them, fine-tune
c) Concatenate ensemble members, jointly prune, fine-tune.
The results here are rather close. We make the following observations:

b) Not beneficial compared to a)
a) Is parallelizable and needs the least amount of memory but has slightly worse results than c).
c) Best option.

Run e.g. with python experiments/ex6_order.py "m_p_ft" 0
First param: method (m_p_ft, m_ft_p_ft, p_m_ft)
Second param: random seed
'''





import copy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from fusion_methods.distillation import train_distill_ensemble
import torch_pruning as tp
from torch import nn
from torch.optim.lr_scheduler import ConstantLR, MultiStepLR

from train import train_model

from fusion_methods.neuron_transplantation import fuse_ensemble, concat_models
from fusion_methods.optimal_transport import fuse_optimal_transport
from fusion_methods.classical_ensembles import evaluate_accuracy_output_averaging
from fusion_methods.weight_averaging import average_weights

from train_helper import set_all_seeds, evaluate_model
import torch
from model import get_model_from_name
from dataloader import get_dataloader_from_name

if __name__ == "__main__":

    # command line arguments: dataset, model(dataset), seed

    # argv[0] is the file name ex11_comparison.py
    method = sys.argv[1]  # p_m_ft , m_p_ft, m_ft_p_ft,
    master_seed = int(sys.argv[2])  # 0,1,2,3...
    num_models = 2

    print(method, master_seed)

    b = 256
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.

    set_all_seeds(master_seed)  # master seed
    train_loader, valid_loader, test_loader = get_dataloader_from_name("cifar10", root="cifar10", batch_size=b)

    image_shape = None
    example_inputs = None
    for images, labels in train_loader:
        image_shape = images.shape
        example_inputs = images
        break
    example_inputs.to("cpu")

    # get models
    models = []
    models_found = False

    try:
        for i in range(num_models):
            model_dict = torch.load(f'models/{"smallnn"}_nobias_{"cifar10"}_{master_seed}_{i}.pt', map_location=device)
            input_dim = image_shape[1] * image_shape[2] * image_shape[3]
            model = get_model_from_name("smallnn", output_dim=num_classes, input_dim=input_dim,
                                        dataset_name="cifar10", bias=False)
            model.load_state_dict(model_dict)
            models.append(model)

    except FileNotFoundError:
        print("models not found")
        exit()

    # model is trained in experiment 11. Run python experiments/ex11_train.py "cifar10" "smallnn" 0   for example.

    accuracy_history = []  # tracks the accuracy of the current method!
    combined_model = None
    set_all_seeds(master_seed)
    train_loader, valid_loader, test_loader = get_dataloader_from_name("cifar10", batch_size=b, root="cifar10")

    if method == "m_p_ft" or method == "m_ft_p_ft":

        for model in models:
            model.to("cpu")

        # merge
        combined_model = concat_models(models)
        combined_model.to(device)

        for model in models:
            model.to(device)

        loss, accuracy = evaluate_model(combined_model, test_loader, device)
        print(accuracy)
        accuracy_history.append(accuracy)

        if method == "m_ft_p_ft":
            # do 3 epochs finetuning here
            optimizer = torch.optim.SGD(combined_model.parameters(), lr=learning_rate, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=3)
            test_acc_list_finetune = train_model(model=combined_model, optimizer=optimizer, criterion=criterion,
                                                 scheduler=scheduler,
                                                 train_loader=train_loader, valid_loader=test_loader, e=3,
                                                 device=device)
            accuracy_history = accuracy_history + test_acc_list_finetune

        combined_model.to("cpu")

        # prune
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner_combined = tp.pruner.MagnitudePruner(
            combined_model,
            example_inputs,
            importance=imp,
            pruning_ratio=0.5,
            root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
            ignored_layers=[combined_model.fc4],
        )
        pruner_combined.step()

    if method == "p_m_ft":

        # prune models individually
        for model in models:
            model.to("cpu")
            imp = tp.importance.MagnitudeImportance(p=2)
            pruner = tp.pruner.MagnitudePruner(
                model,
                example_inputs,
                importance=imp,
                pruning_ratio=0.5,
                root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
                ignored_layers=[model.fc4],
            )
            pruner.step()

        # merge
        combined_model = concat_models(models)
        combined_model.to(device)

    # go back to gpu to finetune
    combined_model.to(device)

    optimizer = torch.optim.SGD(combined_model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=20)
    test_acc_list_finetune = train_model(model=combined_model, optimizer=optimizer, criterion=criterion,
                                         scheduler=scheduler,
                                         train_loader=train_loader, valid_loader=test_loader, e=20,
                                         device=device)
    accuracy_history = accuracy_history + test_acc_list_finetune

    torch.save(accuracy_history, f"out/ex6_{method}_{master_seed}_acc.pt")
