'''
This experiment is a sanity check. We test whether layer-wise vertical concatenation works by comparing it to
output averaging (which should be equivalent since we average the output layers as well).
'''


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from fusion_methods.neuron_transplantation import concat_models
from fusion_methods.classical_ensembles import evaluate_accuracy_output_averaging
from train import train_model
from dataloader import get_dataloaders_cifar10
from train_helper import set_all_seeds, evaluate_model
from model import NeuralNetwork

if __name__ == '__main__':
    print("running experiment 2")

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
    for images, labels in train_loader:
        image_shape = images.shape
        break

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
            print("found models from experiment 2")

        except FileNotFoundError:
            print("experiment 2 models not found")

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

    for i, model in enumerate(models):
        model.eval()
        _, test_accuracy = evaluate_model(model, test_loader, device)
        print(f"model {i} test accuracy: {test_accuracy}")

    # combine and compare the combined model to output averaging!

    # build large model:

    # combined_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    # concat_models(models[0], models[1], models[2], models[3], combined_model)
    for model in models:
        model.to("cpu")
    combined_model = concat_models(models)
    combined_model.to(device)
    for model in models:
        model.to(device)

    # evaluate output averaging
    set_all_seeds(0)
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="cifar",
        validation_fraction=0.1,
        train_transforms=cifar_10_transforms,
        test_transforms=cifar_10_transforms,
        num_workers=0
    )
    acc = evaluate_accuracy_output_averaging(test_loader, models, device)
    print("output averaging: ", acc)
    # 0.6233999729156494

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
    # 0.6233999729156494

    # compare logits
    for i, (vinputs, vlabels) in enumerate(test_loader):
        vinputs = vinputs.to(device)

        combined_model.eval()
        predictions_combined = combined_model(vinputs)  # Calculate model output.

        # predictions_output_averaging
        predictions = []  # shape (model_num, batchsize, classes)
        for model in models:
            model.to(device)
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                output = model(vinputs)
            predictions.append(output)
        aggregated_predictions = torch.stack(predictions).mean(dim=0)  # (calc mean --> shape (batchsize,classes) )
        print("prediction tensors are equal?: ",
              torch.allclose(predictions_combined, aggregated_predictions, atol=1e-2))
        break

    for module in combined_model.modules():
        print(module)
