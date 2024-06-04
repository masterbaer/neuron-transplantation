'''
Experiment 11 compares Neuron Transplantation with
a) Optimal Transport Fusion (from Singh and Jaggi),
b) Vanilla Averaging
c) Selecting the best Ensemble Member
d) Output Averaging
under fine-tuning and distillation (the fused model is used as the student model and
the ensemble with output averaging as the teacher).
'''

import copy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from fusion_methods.distillation import train_distill_ensemble

from torch import nn
from torch.optim.lr_scheduler import ConstantLR, MultiStepLR

from train import train_model

from fusion_methods.neuron_transplantation import fuse_ensemble
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
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    master_seed = int(sys.argv[3])
    method = sys.argv[4]
    # argmax, full_ensemble, avg_ft, nt_ft, ot_ft, model0_distill, avg_distill, nt_distill, ot_distill

    num_models = 2
    #num_models = 4

    try:
        acc = torch.load(f"out/ex11_{model_name}_nobias_{dataset_name}_{master_seed}_{method}_acc.pt")
        print("experiment already done")
        exit()
    except FileNotFoundError:
        pass

    b = 256
    e = 30
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10
    if dataset_name == "cifar100":
        num_classes = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.

    set_all_seeds(master_seed)  # master seed
    train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, root=dataset_name, batch_size=b)

    image_shape = None
    example_inputs = None
    for images, labels in train_loader:
        image_shape = images.shape
        example_inputs = images
        break
    example_inputs.to(device)

    # get models
    models = []
    models_found = False

    try:
        for i in range(num_models):
            model_dict = torch.load(f'models/{model_name}_nobias_{dataset_name}_{master_seed}_{i}.pt', map_location=device)
            input_dim = image_shape[1] * image_shape[2] * image_shape[3]
            model = get_model_from_name(model_name, output_dim=num_classes, input_dim=input_dim,
                                        dataset_name=dataset_name, bias=False)
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
            torch.save(best_individual_acc,
                       f"out/ex11_{model_name}_nobias_{dataset_name}_{master_seed}_{method}_acc.pt")

    if method == "full_ensemble":
        set_all_seeds(master_seed)
        train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, batch_size=b,
                                                                           root=dataset_name)
        acc_ensemble = evaluate_accuracy_output_averaging(test_loader, models, device)
        print("full ensemble acc: ", acc_ensemble)
        torch.save(acc_ensemble,
                   f"out/ex11_{model_name}_nobias_{dataset_name}_{master_seed}_{method}_acc.pt")

    # test the methods under finetuning
    if method == "avg_ft" or method == "nt_ft" or method == "ot_ft":

        for model in models:
            model.to("cpu")
        example_inputs.to("cpu")
        fused_model = None

        if method == "avg_ft":
            fused_model = average_weights(models)

        if method == "nt_ft":
            fused_model = fuse_ensemble(models, example_inputs)

        if method == "ot_ft":
            fused_model = fuse_optimal_transport(models)

        set_all_seeds(master_seed)
        train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, batch_size=b, root=dataset_name)
        fused_model.to(device)

        loss, accuracy = evaluate_model(fused_model, test_loader, device)
        print(accuracy)
        test_acc_list = [accuracy]

        optimizer = torch.optim.SGD(fused_model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)

        if model_name == "resnet18" or model_name == "√it":
            # https://arxiv.org/pdf/1910.05653.pdf used almost the setting below (they trained 300epochs with milestones
            # 150,250 instead and used the best checkpoint instead of the last

            learning_rate = 0.1
            e = 100
            momentum = 0.9
            optimizer = torch.optim.SGD(fused_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
            milestones = [50, 80]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        test_acc_list_finetune = train_model(model=fused_model, optimizer=optimizer, criterion=criterion,
                                             scheduler=scheduler,
                                             train_loader=train_loader, valid_loader=test_loader, e=e,
                                             device=device)
        test_acc_list = test_acc_list + test_acc_list_finetune

        torch.save(test_acc_list,
                   f"out/ex11_{model_name}_nobias_{dataset_name}_{master_seed}_{method}_acc.pt")

    # test the methods under distillation
    if method == "model0_distill" or method == "avg_distill" or method == "nt_distill" or method == "ot_distill":
        fused_model = None

        if method == "model0_distill":
            fused_model = copy.deepcopy(models[0])

        if method == "avg_distill":
            fused_model = average_weights(models)

        if method == "nt_distill":
            fused_model = fuse_ensemble(models, example_inputs)

        if method == "ot_distill":
            fused_model = fuse_optimal_transport(models)

        soft_target_loss_weight = 1  # train with only the ensemble logits
        T = 2
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

        if model_name == "resnet18" or model_name == "√it":
            # https://arxiv.org/pdf/1910.05653.pdf used almost the setting below (they trained 300epochs with milestones
            # 150,250 instead and used the best checkpoint instead of the last

            learning_rate = 0.1
            e = 100
            momentum = 0.9
            optimizer = torch.optim.SGD(fused_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
            milestones = [50, 80]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        test_acc_list_distill = train_distill_ensemble(student=fused_model, models=models,
                                                       train_loader=train_loader,
                                                       valid_loader=test_loader, optimizer=optimizer,
                                                       criterion=criterion,
                                                       scheduler=scheduler, e=e, T=T,
                                                       soft_target_loss_weight=soft_target_loss_weight,
                                                       device=device)
        test_acc_list = test_acc_list + test_acc_list_distill

        torch.save(test_acc_list,
                   f"out/ex11_{model_name}_nobias_{dataset_name}_{master_seed}_{method}_acc.pt")

