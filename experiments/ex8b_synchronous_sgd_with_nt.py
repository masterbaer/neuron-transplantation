'''
Part two of the self-fusion. Here we fuse "similar" models with each other, in particular we use NT as a synchronization
method for parallel SGD. This optimization method fails to train. We attribute this to the information loss from fusing
almost identical models. It might be possible to artificially create diversity
(comparable to Sun et. al in Ensemble Compression DOI:10.1007/978-3-319-71249-9_12) to make this work.
'''

import sys
from pathlib import Path
from mpi4py import MPI
import torch
import torchvision

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fusion_methods.neuron_transplantation import fuse_ensemble
from fusion_methods.weight_averaging import average_weights
from dataloader import get_dataloader_from_name
from train_helper import set_all_seeds, evaluate_model
from model import NeuralNetwork

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    seed = rank
    set_all_seeds(seed)
    method = sys.argv[1]
    b = 256
    e = 60
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    set_all_seeds(seed)  # Set all seeds to chosen random seed.
    print("seed:", seed, "using device:", device, "world_size:", world_size)

    # every worker has the full dataset for now
    train_loader, valid_loader, test_loader = get_dataloader_from_name("svhn", root="svhn", batch_size=b)

    example_inputs = None
    image_shape = None
    for images, labels in train_loader:
        image_shape = images.shape
        example_inputs = images
        example_inputs.to("cpu")
        break

    # each worker trains 1 neural network
    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    test_accuracy_list = []

    for epoch in range(e):
        model.train()

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

        # evaluate individual models locally
        model.eval()
        _, test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracy_list.append(test_accuracy)
        if rank == 0:
            print(f"test accuracy after epoch {epoch}: {test_accuracy}")

        # collect models and average / nt-fuse them
        state_dict = model.state_dict()
        all_state_dicts = comm.allgather(state_dict)

        models = []
        for i in range(world_size):
            collected_model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
            collected_model.load_state_dict(all_state_dicts[i])
            collected_model.to("cpu")
            models.append(collected_model)

        fused_model = None

        if method == "nt":
            fused_model = fuse_ensemble(models, example_inputs)
        if method == "avg":
            fused_model = average_weights(models)

        model.load_state_dict(fused_model.state_dict())
        model.to(device)

    if rank == 0:
        torch.save(test_accuracy_list, f'out/ex8b_accuracies_{method}.pt')

# avg:
# test accuracy after epoch 0: 0.19587430357933044
# test accuracy after epoch 59: 0.8165718913078308

# nt: fails to train
# test accuracy after epoch 0: 0.19587430357933044
# test accuracy after epoch 59: 0.19587430357933044
