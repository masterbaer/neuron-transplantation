'''
Simple sanity check. We test Neuron Transplantation on the Convolutional Neurol Networks
a) LeNet b) VGG11 c) Resnet18 (to change between them change the get_model_by_name() method.
We do not train or fine-tune the models. This is only a syntactical sanity check
'''



import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torchvision import transforms

from fusion_methods.neuron_transplantation import concat_models, isLeafLayer, \
    getLastLayerName, get_module_by_name
import torch_pruning as tp
from dataloader import get_dataloaders_cifar10
from train_helper import set_all_seeds
from model import LeNet, get_model_from_name

if __name__ == '__main__':

    b = 256
    e = 60
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.

    # https://github.com/soapisnotfat/pytorch-cifar10/blob/master/main.py
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="cifar",
        validation_fraction=0.1,
        train_transforms=train_transform,
        test_transforms=test_transform,
        num_workers=0
    )

    image_shape = None
    example_inputs = None
    for images, labels in train_loader:
        image_shape = images.shape
        example_inputs = images
        break
    example_inputs.to(device)
    input_dim = image_shape[1] * image_shape[2] * image_shape[3]
    models = []
    for i in range(4):
        set_all_seeds(i)

        # try out the other two variants
        # model = LeNet().to(device)
        # model = VGG("VGG11", num_classes).to(device)
        # model = ResNet18().to(device)
        model = get_model_from_name("resnet18", input_dim=input_dim, output_dim=10, bias=False)

        models.append(model)

    # print original architecture:
    for name, module in models[0].named_modules():
        if isLeafLayer(module):
            print(module)
    print("-----------------------------------------------")

    #combined_model = fuse_ensemble(models, example_inputs)

    for model in models:
        model.to("cpu")
    combined_model = concat_models(models)
    combined_model.to(device)
    for model in models:
        model.to(device)

    # print layers
    print("Concatenated model architecture: ")
    for name, module in combined_model.named_modules():
        if isLeafLayer(module):
            print(module)
    print("-----------------------------------------------")

    combined_parameter_number = len(torch.nn.utils.parameters_to_vector(combined_model.parameters()))

    # prune
    combined_model.to("cpu")
    example_inputs.to("cpu")

    last_layer_name = getLastLayerName(combined_model)
    last_layer = get_module_by_name(combined_model, last_layer_name)

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        combined_model,
        example_inputs,
        importance=imp,
        pruning_ratio=0.75,
        root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
        ignored_layers=[last_layer],
    )
    pruner.step()

    combined_model.to(device)

    print("Fused Architecture: ")
    for module in combined_model.modules():
        if isLeafLayer(module):
            print(module)

    print("parameters of single model: ", len(torch.nn.utils.parameters_to_vector(models[0].parameters())))
    print("combined model parameters: ", combined_parameter_number)
    print("fused model parameters: ", len(torch.nn.utils.parameters_to_vector(combined_model.parameters())))
