'''
This file is to train some (4) small models quickly in parallel and compare the individual models to
a) output averaging of the ensemble members
b) large model with capacity of the ensemble.
Note that training these models is not relevant to reproduce the other experiments in case mpi4py is troublesome to run
'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mpi4py import MPI
import torch
import torchvision

from fusion_methods.classical_ensembles import evaluate_accuracy_output_averaging
from dataloader import get_dataloaders_cifar10
from train_helper import set_all_seeds, evaluate_model
from model import NeuralNetwork, CombinedNeuralNetwork

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    seed = rank
    set_all_seeds(seed)

    b = 256
    e = 60
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    set_all_seeds(seed)  # Set all seeds to chosen random seed.
    print("seed:", seed, "using device:", device, "world_size:", world_size)

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

    # each worker trains 1 neural network
    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

    # This is the big model again, but it is trained from scratch instead of being combined from the ensemble members.
    combined_model_scratch = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_combined_scratch = torch.optim.SGD(combined_model_scratch.parameters(), lr=learning_rate, momentum=momentum)

    combined_test_accuracy_list = []
    combined_scratch_test_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(e):

        model.train()
        if rank == 0:
            combined_model_scratch.train()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # train the local models
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # train the combined model from scratch
            if rank == 0:
                optimizer_combined_scratch.zero_grad()
                outputs_combined_scratch = combined_model_scratch(inputs)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss_combined_scratch = loss_fn(outputs_combined_scratch, labels)
                loss_combined_scratch.backward()
                optimizer_combined_scratch.step()

        # evaluate individual models locally
        model.eval()
        _, test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracy_list.append(test_accuracy)

        # evaluate large model trained from scratch on rank 0
        # The large model sees all data comparable to an individual model.
        if rank == 0:
            combined_model_scratch.eval()
            _, combined_scratch_test_accuracy = evaluate_model(combined_model_scratch, test_loader, device)
            combined_scratch_test_accuracy_list.append(combined_scratch_test_accuracy)

            print(f"test accuracy after {epoch+1} epochs: small and large model:{test_accuracy, combined_scratch_test_accuracy}")

        # collect models at rank 0 to evaluate the ensemble performance
        state_dict = model.state_dict()
        state_dict = comm.gather(state_dict, root=0)
        if rank == 0:
            # evaluate the ensemble accuracy
            model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model0.load_state_dict(state_dict[0])
            model1.load_state_dict(state_dict[1])
            model2.load_state_dict(state_dict[2])
            model3.load_state_dict(state_dict[3])
            combined_test_accuracy = evaluate_accuracy_output_averaging(test_loader,
                                                                        [model0, model1, model2, model3], device)
            combined_test_accuracy_list.append(combined_test_accuracy)
            print("ensemble accuracy: ", combined_test_accuracy)

    torch.save(model.state_dict(), f"models/ex1_model_{rank}.pt")
    torch.save(test_accuracy_list, f'out/ex1_test_accuracy{rank}.pt')

    if rank == 0:
        torch.save(combined_scratch_test_accuracy_list, 'out/ex1_combined_scratch_test_accuracy.pt')
        torch.save(combined_test_accuracy_list, 'out/ex1_combined_test_accuracy.pt')
        torch.save(combined_model_scratch.state_dict(), "out/ex1_combined_model_scratch.pt")
