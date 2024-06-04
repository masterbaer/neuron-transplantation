'''
Here we use our model fusion technique "Neuron Transplantation". We first create a vertically concatenated model of all
ensemble members before pruning it down to original size.
Run with "python experiments/ex4_fusion.py"
'''

import sys
from pathlib import Path

from matplotlib import pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from fusion_methods.neuron_transplantation import concat_models
from train import train_model
import torch_pruning as tp
from dataloader import get_dataloaders_cifar10
from train_helper import set_all_seeds, evaluate_model
from model import NeuralNetwork

if __name__ == '__main__':
    print("running experiment 4 (trying neuron transplantation with pruning and finetuning) ")

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

    # check whether 4 models are available from experiment 1.
    if not models_found:
        try:

            dict0 = torch.load("models/ex1_model_0.pt", map_location=device)
            dict1 = torch.load("models/ex1_model_1.pt", map_location=device)
            dict2 = torch.load("models/ex1_model_2.pt", map_location=device)
            dict3 = torch.load("models/ex1_model_3.pt", map_location=device)

            model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

            model0.load_state_dict(dict0)
            model1.load_state_dict(dict1)
            model2.load_state_dict(dict2)
            model3.load_state_dict(dict3)

            models.append(model0)
            models.append(model1)
            models.append(model2)
            models.append(model3)

            models_found = True
            print("found models from experiment 1")
        except FileNotFoundError:
            print("experiment 1 models not found")

    # check whether 4 models are available from experiment 2.
    if not models_found:
        try:
            dict0 = torch.load(f"models/smallnn_model_cifar10_0", map_location=device)
            dict1 = torch.load(f"models/smallnn_model_cifar10_1", map_location=device)
            dict2 = torch.load(f"models/smallnn_model_cifar10_2", map_location=device)
            dict3 = torch.load(f"models/smallnn_model_cifar10_3", map_location=device)

            model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

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

            model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)
            # train model
            val_acc_list = train_model(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                       train_loader=train_loader, valid_loader=valid_loader, e=e, device=device)
            torch.save(model.state_dict(), f'models/smallnn_model_cifar10_{i}')
            models.append(model)

    print("checking if models are trained by evaluating them: ")

    best_individual_acc = 0.0
    for i, model in enumerate(models):
        model.eval()
        _, test_accuracy = evaluate_model(model, test_loader, device)
        if test_accuracy > best_individual_acc:
            best_individual_acc = test_accuracy
        print(f"model {i} test accuracy: {test_accuracy}")


    # combine and compare the combined model to output averaging!

    # build large model:
    for model in models:
        model.to("cpu")
    combined_model = concat_models(models)
    combined_model.to(device)
    for model in models:
        model.to(device)

    for module in combined_model.modules():
        print(module)

    # evaluate combined model
    set_all_seeds(0)
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="cifar",
        validation_fraction=0.1,
        train_transforms=cifar_10_transforms,
        test_transforms=cifar_10_transforms,
        num_workers=0
    )
    _, acc_combined = evaluate_model(combined_model, test_loader, device)
    print("combined model: ", acc_combined)
    # model 0 test accuracy: 0.5665000081062317
    # model 1 test accuracy: 0.5654000043869019
    # model 2 test accuracy: 0.5595999956130981
    # model 3 test accuracy: 0.5667999982833862

    # combined: 0.6233999729156494

    # prune into original shape and finetune the model !

    # go to cpu to prune the model with torch-pruning
    combined_model.to("cpu")
    example_inputs.to("cpu")

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner_combined = tp.pruner.MagnitudePruner(
        combined_model,
        example_inputs,
        importance=imp,
        pruning_ratio=0.75,
        root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
        ignored_layers=[combined_model.fc4],
    )
    pruner_combined.step()

    # go back to gpu to finetune
    combined_model.to(device)

    _, acc_fused = evaluate_model(combined_model, test_loader, device)
    print("combined model after pruning: ", acc_fused)
    # pruned: 0.16509999334812164

    print("architecture after pruning:")
    for name, layer in combined_model.named_modules():
        print(layer)

    # Flatten(start_dim=1, end_dim=-1)
    # Linear(in_features=12288, out_features=512, bias=True)
    # Linear(in_features=512, out_features=512, bias=True)
    # Linear(in_features=512, out_features=512, bias=True)
    # Linear(in_features=512, out_features=10, bias=True)

    # finetuning:

    print("finetuning the pruned model")

    set_all_seeds(0)
    e = 30

    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="cifar",
        validation_fraction=0.1,
        train_transforms=cifar_10_transforms,
        test_transforms=cifar_10_transforms,
        num_workers=0
    )

    optimizer = torch.optim.SGD(combined_model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)
    # train model

    # giving the test loader here to get the test accuracies
    test_acc_list = train_model(model=combined_model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                               train_loader=train_loader, valid_loader=test_loader, e=e, device=device)

    # finetuned:
    # valid acc in epoch 1: 0.5683000087738037
    # valid acc in epoch 2: 0.576200008392334
    # valid acc in epoch 3: 0.5843999981880188
    # valid acc in epoch 4: 0.5817999839782715
    # valid acc in epoch 5: 0.5859999656677246
    # valid acc in epoch 6: 0.5824999809265137
    # valid acc in epoch 7: 0.585099995136261
    # valid acc in epoch 8: 0.5907999873161316
    # valid acc in epoch 9: 0.5877999663352966
    # valid acc in epoch 10: 0.5878999829292297

    # model 0 test accuracy: 0.5665000081062317
    # model 1 test accuracy: 0.5654000043869019
    # model 2 test accuracy: 0.5595999956130981
    # model 3 test accuracy: 0.5667999982833862

    # combined: 0.6233999729156494

    # --> some of the ensemble accuracy is recovered

    plt.plot(test_acc_list, label="finetuning of combined model", marker="o")
    plt.plot([acc_combined], label="full ensemble accuracy", marker="s")
    plt.axhline(y=best_individual_acc, color='b', linestyle=':')

    plt.legend()
    plt.xlabel("finetuning epochs")
    plt.ylabel("accuracy")
    plt.savefig(f"plots/ex4_fusion.png")
