'''
Quick sanity check of vertical concatenation of models with different layer sizes.
Run with "python experiments/ex3.py"
'''

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from fusion_methods.neuron_transplantation import concat_models
from train import train_model
from dataloader import get_dataloaders_cifar10
from train_helper import set_all_seeds, evaluate_model
from model import AdaptiveNeuralNetwork

if __name__ == '__main__':
    print("running experiment 3 (trying neuron transplantation with different sizes) ")

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

    models = []

    models_found = False

    # check whether 4 models are available from experiment 1.
    if not models_found:
        try:

            dict0 = torch.load("models/ex3_model_0.pt", map_location=device)
            dict1 = torch.load("models/ex3_model_1.pt", map_location=device)
            dict2 = torch.load("models/ex3_model_2.pt", map_location=device)
            dict3 = torch.load("models/ex3_model_3.pt", map_location=device)

            model0 = AdaptiveNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                                           256, 256, 256).to(device)
            model1 = AdaptiveNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                                           128, 64, 64).to(device)
            model2 = AdaptiveNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                                           64, 128, 64).to(device)
            model3 = AdaptiveNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                                           64, 64, 128).to(device)

            model0.load_state_dict(dict0)
            model1.load_state_dict(dict1)
            model2.load_state_dict(dict2)
            model3.load_state_dict(dict3)

            models.append(model0)
            models.append(model1)
            models.append(model2)
            models.append(model3)

            models_found = True
            print("found models from experiment 3")
        except FileNotFoundError:
            print("experiment 3 models not found")

    # check whether 3 models are available from experiment 2.

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

            # create some models with different hidden sizes
            if i == 0:
                model = AdaptiveNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                                              256, 256, 256).to(device)
            if i == 1:
                model = AdaptiveNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                                              128, 64, 64).to(device)
            if i == 2:
                model = AdaptiveNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                                              64, 128, 64).to(device)
            if i == 3:
                model = AdaptiveNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                                              64, 64, 128).to(device)

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)
            # train model
            val_acc_list = train_model(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                       train_loader=train_loader, valid_loader=valid_loader, e=e, device=device)
            torch.save(model.state_dict(), f'models/ex3_model_{i}.pt')
            models.append(model)

    print("checking if models are trained by evaluating them: ")

    for i, model in enumerate(models):
        model.eval()
        _, test_accuracy = evaluate_model(model, test_loader, device)
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

    # model 0 test accuracy: 0.5565999746322632
    # model 1 test accuracy: 0.5446000099182129
    # model 2 test accuracy: 0.528499960899353
    # model 3 test accuracy: 0.526699960231781

    # AdaptiveNeuralNetwork(
    #   (flatten): Flatten(start_dim=1, end_dim=-1)
    #   (fc1): Linear(in_features=12288, out_features=512, bias=True)
    #   (fc2): Linear(in_features=512, out_features=512, bias=True)
    #   (fc3): Linear(in_features=512, out_features=512, bias=True)
    #   (fc4): Linear(in_features=512, out_features=10, bias=True)
    # )

    # combined model:  0.5881999731063843
