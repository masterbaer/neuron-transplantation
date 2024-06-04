'''
Sanity check to see if distillation works.
'''
import copy
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


if __name__ == '__main__':
    print("trying knowledge distillation")

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

    acc_ensemble = evaluate_accuracy_output_averaging(test_loader, models, device)
    print("full ensemble acc: ", acc_ensemble)

    # do ensemble distillation
    set_all_seeds(0)  # todo maybe use a different seed (than that of model0) ?
    #student = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    student = copy.deepcopy(models[0])
    # perhaps try the fused model as the student similar to https://arxiv.org/pdf/1910.05653.pdf
    # perhaps try the normal averaged mode as well (which they did not do) as the student

    # https://arxiv.org/pdf/1910.05653.pdf: (e.g. table S14):
    # they have a look at A-->B distillation whereas A is a big model and B
    # is a small model (or an ensemble of small models that they average)
    # unfortunately they do not compare OT to distillation as a fusion technique

    e = 30

    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="cifar",
        validation_fraction=0.1,
        train_transforms=cifar_10_transforms,
        test_transforms=cifar_10_transforms,
        num_workers=0
    )

    optimizer = torch.optim.SGD(student.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)

    # how much of the "original" loss and how much of "soft target" loss is being kept

    # for further experiments tune these hyperparameters
    #soft_target_loss_weight = 0.25
    soft_target_loss_weight = 1.0
    T = 2

    # giving the test loader here to get the test accuracies
    test_acc_list = train_distill_ensemble(student=student, models=models, train_loader=train_loader,
                                           valid_loader=test_loader, optimizer=optimizer, criterion=criterion,
                                           scheduler=scheduler, e=e, T=T,
                                           soft_target_loss_weight=soft_target_loss_weight, device=device)

    plt.plot(test_acc_list, label="distill accuracy", marker="o")
    plt.plot([acc_ensemble], label="full ensemble accuracy", marker="s")
    plt.axhline(y=best_individual_acc, color='b', linestyle=':')

    plt.legend()
    plt.xlabel("distillation epochs")
    plt.ylabel("test accuracy")
    plt.savefig(f"plots/ex10b_distillation.png")

    # T = 2, soft_target_loss_weight = 1
    #model 0 test accuracy: 0.574400007724762
    #model 1 test accuracy: 0.5666999816894531
    #model 2 test accuracy: 0.5676999688148499
    #model 3 test accuracy: 0.5701000094413757
    #full ensemble acc:  0.6225999593734741

    #valid acc in epoch 1: 0.5874999761581421
    #valid acc in epoch 2: 0.5820000171661377
    #valid acc in epoch 3: 0.5848000049591064
    #valid acc in epoch 4: 0.588100016117096
    #valid acc in epoch 5: 0.5968999862670898
    #valid acc in epoch 6: 0.593500018119812
    #valid acc in epoch 7: 0.5877000093460083
