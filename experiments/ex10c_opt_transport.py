'''
Sanity check to see if Optimal Transport Fusion from Singh and Jaggi works (https://arxiv.org/pdf/1910.05653.pdf).
'''
import sys
from pathlib import Path

from matplotlib import pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from train import train_model
from dataloader import get_dataloaders_cifar10
from train_helper import set_all_seeds, evaluate_model
from model import NeuralNetwork
from fusion_methods.distillation import train_distill_ensemble
from fusion_methods.classical_ensembles import evaluate_accuracy_output_averaging
from fusion_methods.optimal_transport import fuse_optimal_transport

if __name__ == '__main__':
    print("trying optimal transport")

    b = 256
    e = 60
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.

    cifar_10_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Get PYTORCH dataloaders for training, testing, and validation dataset.
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="cifar",
        validation_fraction=0.1,
        train_transforms=cifar_10_transforms,
        test_transforms=cifar_10_transforms,
        num_workers=0
    )

    image_shape = None
    example_inputs = None
    for images, labels in train_loader:
        image_shape = images.shape
        example_inputs = images
        break
    example_inputs.to(device)

    models = []

    models_found = False

    # check whether 4 models are available.
    if not models_found:
        try:
            dict0 = torch.load(f"models/smallnn_nobias_model_cifar10_0", map_location=device)
            dict1 = torch.load(f"models/smallnn_nobias_model_cifar10_1", map_location=device)
            dict2 = torch.load(f"models/smallnn_nobias_model_cifar10_2", map_location=device)
            dict3 = torch.load(f"models/smallnn_nobias_model_cifar10_3", map_location=device)

            model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes, bias=False).to(device)
            model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes, bias=False).to(device)
            model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes, bias=False).to(device)
            model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes, bias=False).to(device)

            model0.load_state_dict(dict0)
            model1.load_state_dict(dict1)
            model2.load_state_dict(dict2)
            model3.load_state_dict(dict3)

            models.append(model0)
            models.append(model1)
            models.append(model2)
            models.append(model3)

            models_found = True
            print("found models")

        except FileNotFoundError:
            print("models not found")

    if not models_found:
        # train the models and save them
        print("training models")

        for i in range(4):
            set_all_seeds(i)

            train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
                batch_size=b,
                root="cifar",
                validation_fraction=0.1,
                train_transforms=cifar_10_transforms,
                test_transforms=cifar_10_transforms,
                num_workers=0
            )

            model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes, bias=False).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)
            # train model
            val_acc_list = train_model(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                       train_loader=train_loader, valid_loader=valid_loader, e=e, device=device)
            torch.save(model.state_dict(), f'models/smallnn_nobias_model_cifar10_{i}')
            models.append(model)

    # remove biases from the models:
    #for model in models:
    #    for layer in model.modules():
    #        if isinstance(layer, nn.Linear):
    #            layer.bias = None

    set_all_seeds(0)

    print("checking if models are trained by evaluating them: ")

    best_individual_acc = 0.0
    for i, model in enumerate(models):
        model.eval()
        _, test_accuracy = evaluate_model(model, test_loader, device)
        if test_accuracy > best_individual_acc:
            best_individual_acc = test_accuracy
        print(f"model {i} test accuracy: {test_accuracy}")

    acc_ensemble = evaluate_accuracy_output_averaging(test_loader, models, device)
    print("full ensemble acc: ", acc_ensemble)


    # OT adapted from https://github.com/sidak/otfusion/blob/master/distillation_big_only.py

    for model in models:
        model.to("cpu")

    set_all_seeds(0)
    fused_model = fuse_optimal_transport(models, train_loader, test_loader, device)

    fused_model.to(device)
    loss, accuracy = evaluate_model(fused_model, test_loader, device)
    print(accuracy)
    test_acc_list = [accuracy]

    e = 30
    optimizer = torch.optim.SGD(fused_model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)
    test_acc_list_ = train_model(model=fused_model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                train_loader=train_loader, valid_loader=test_loader, e=e, device=device)
    test_acc_list = test_acc_list + test_acc_list_

    plt.plot(test_acc_list, label="finetuning of fused model", marker="o")
    plt.plot([acc_ensemble], label="full ensemble accuracy", marker="s")
    plt.axhline(y=best_individual_acc, color='b', linestyle=':')

    plt.legend()
    plt.xlabel("finetuning epochs")
    plt.ylabel("accuracy")
    plt.savefig(f"plots/ex10c_ot.png")
