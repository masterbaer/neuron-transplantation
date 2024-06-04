'''
Here we check the fusion of a model with itself. The accuracy drops a little and can be recovered quickly.
But there is no point in doing so as half of the neurons are deleted in the process.
'''

import copy

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from torch.optim.lr_scheduler import ConstantLR

from dataloader import get_dataloader_from_name
from fusion_methods.neuron_transplantation import fuse_ensemble
from model import AdaptiveNeuralNetwork2
from train import train_model
from train_helper import set_all_seeds, evaluate_model

# train 1 model only (checkpoint the best iteration) and then merge it with itself and see how the performance drops
# and rises again

if __name__ == "__main__":

    dataset_name = "svhn"
    model_width = 512
    model_depth = 4

    master_seed = 0

    b = 256
    e = 100
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.

    found_model = False
    checkpoint_model_dict = None
    try:
        checkpoint_model_dict = torch.load(f'models/ex8a_model.pt', map_location=device)
        print("trained model already exists")
        found_model = True
    except FileNotFoundError:
        found_model = False
        print("model not found")

    if not found_model:
        set_all_seeds(master_seed)

        train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, root=dataset_name,
                                                                       batch_size=b)
        image_shape = None
        for images, labels in train_loader:
            image_shape = images.shape
            example_inputs = images
            break

        input_dim = image_shape[1] * image_shape[2] * image_shape[3]

        model = AdaptiveNeuralNetwork2(input_dim=input_dim, output_dim=num_classes,
                                   layer_width=model_width, num_layers=model_depth).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)
        checkpoint_model = copy.deepcopy(model)

        # train model
        val_acc_list = train_model(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                train_loader=train_loader, valid_loader=valid_loader, e=e, device=device,
                                checkpoint=True, checkpoint_model=checkpoint_model)
        checkpoint_model_dict = checkpoint_model.state_dict()
        torch.save(checkpoint_model_dict, f'models/ex8a_model.pt')

    # do self-nt and compare it to the original accuracy

    set_all_seeds(master_seed)

    train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, root=dataset_name,
                                                                        batch_size=b)
    image_shape = None
    example_inputs = None
    for images, labels in train_loader:
        image_shape = images.shape
        example_inputs = images
        break

    input_dim = image_shape[1] * image_shape[2] * image_shape[3]

    model = AdaptiveNeuralNetwork2(input_dim=input_dim, output_dim=num_classes,
                                   layer_width=model_width, num_layers=model_depth).to(device)
    model.load_state_dict(checkpoint_model_dict)
    model.to("cpu")
    fused_model = fuse_ensemble([model, model], example_inputs)

    model.to(device)
    fused_model.to(device)
    # evaluate model accuracy

    _, model_acc = evaluate_model(model, test_loader, device)
    print(model_acc)
    _, fused_acc = evaluate_model(fused_model, test_loader, device)
    print(fused_acc)

    test_acc_list = [fused_acc]
    # finetune

    optimizer = torch.optim.SGD(fused_model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)

    test_acc_list_finetune = train_model(model=fused_model, optimizer=optimizer, criterion=criterion,
                                         scheduler=scheduler,
                                         train_loader=train_loader, valid_loader=test_loader, e=20,
                                         device=device)
    test_acc_list = test_acc_list + test_acc_list_finetune

    torch.save(test_acc_list, f"out/ex8a_nt_acc.pt")

    # 0.830208957195282
    # 0.6763598322868347
    # valid acc in epoch 1: 0.8284419178962708
    # valid acc in epoch 2: 0.8265596032142639
    # valid acc in epoch 3: 0.8259449601173401
    # valid acc in epoch 4: 0.8258297443389893
    # valid acc in epoch 5: 0.8282882571220398
    # valid acc in epoch 6: 0.8280193209648132
    # valid acc in epoch 7: 0.8317839503288269
    # valid acc in epoch 8: 0.8270205855369568
    # valid acc in epoch 9: 0.8250998854637146
    # valid acc in epoch 10: 0.8236401081085205
    # valid acc in epoch 11: 0.8277120590209961
    # valid acc in epoch 12: 0.8256760835647583
    # valid acc in epoch 13: 0.8276352286338806
    # valid acc in epoch 14: 0.822449266910553
    # valid acc in epoch 15: 0.8256376385688782
    # valid acc in epoch 16: 0.8218346238136292
    # valid acc in epoch 17: 0.8237553834915161
    # valid acc in epoch 18: 0.8252151012420654
    # valid acc in epoch 19: 0.809964656829834
    # valid acc in epoch 20: 0.8260602355003357

    # --> accuracy after self-merge is recovered after 1 epoch. But there is no "benefit" in doing this.
    #