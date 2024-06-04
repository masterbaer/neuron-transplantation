import copy
import sys
from pathlib import Path

from torch import nn
from torch.optim.lr_scheduler import ConstantLR, MultiStepLR

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import get_model_from_name
import torch
from dataloader import get_dataloader_from_name
from train import train_model
from train_helper import set_all_seeds

if __name__ == "__main__":

    # command line arguments: dataset, model(dataset), seed

    # argv[0] is the file name ex11_comparison.py
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    master_seed = int(sys.argv[3])
    num_models = 2
    # num_models = 4

    print("dataset: ", dataset_name, "model: ", model_name, "seed: ", master_seed)

    try:
        distill_accs = torch.load(f"out/ex11_{model_name}_nobias_{dataset_name}_{master_seed}_ot_distill_acc.pt")
        print("experiment already done, training models is not necessary")
        exit()
    except FileNotFoundError:
        pass

    b = 256
    e = 60
    learning_rate = 0.01
    momentum = 0.9
    num_classes = 10
    if dataset_name == "cifar100":
        num_classes = 100

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print("training models")

    for i in range(num_models):

        try:
            model_dict = torch.load(f'models/{model_name}_nobias_{dataset_name}_{master_seed}_{i}.pt', map_location=device)
            print("trained model already exists")
        except FileNotFoundError:
            # only train this model if not existent already

            set_all_seeds(100 * master_seed + i)  # local seeds: (0,1,2,3) or (100,101,102,103) if master seed is 0 or 1

            train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, root=dataset_name,
                                                                               batch_size=b)
            image_shape = None
            for images, labels in train_loader:
                image_shape = images.shape
                example_inputs = images
                break

            input_dim = image_shape[1] * image_shape[2] * image_shape[3]
            model = get_model_from_name(model_name=model_name, dataset_name=dataset_name, input_dim=input_dim,
                                        output_dim=num_classes, bias=False).to(device)

            # https://arxiv.org/pdf/1910.05653.pdf used a constant lr of 0.01 for their MLPNet
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=e)

            if model_name == "resnet18" or model_name == "vgg11":

                if dataset_name == "svhn":
                    # see https://github.com/Galaxies99/SVHN-playground/blob/main/configs/ResNet/ResNet18.yaml
                    learning_rate = 0.001
                    batch_size = 128

                if dataset_name == "cifar10":
                    # see https://arxiv.org/pdf/1910.05653.pdf used almost the setting below
                    # S3.1.2
                    batch_size = 128
                    learning_rate = 0.05
                    e = 300  #
                    optimizer_weight_decay = 0.0005
                    optimizer_decay_with_factor = 2.0
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                                                weight_decay=optimizer_weight_decay)
                    milestones = [30, 60, 90, 120, 150, 180, 210, 240, 270]
                    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=1.0/optimizer_decay_with_factor)

                if dataset_name == "cifar100":
                    batch_size = 128
                    learning_rate = 0.05
                    e = 300  #
                    optimizer_weight_decay = 0.0005
                    optimizer_decay_with_factor = 2.0
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                                                weight_decay=optimizer_weight_decay)
                    milestones = [30, 60, 90, 120, 150, 180, 210, 240, 270]
                    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=optimizer_decay_with_factor)



            if model_name == "vgg11" or model_name == "resnet18" or model_name == "vit":
                # train with checkpoint to get the best model
                checkpoint_model = copy.deepcopy(model)
                val_acc_list = train_model(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                           train_loader=train_loader, valid_loader=valid_loader, e=e, device=device,
                                           checkpoint=True, checkpoint_model=checkpoint_model)

                torch.save(checkpoint_model.state_dict(), f'models/{model_name}_nobias_{dataset_name}_{master_seed}_{i}.pt')
            else:
                # train model
                val_acc_list = train_model(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                       train_loader=train_loader, valid_loader=valid_loader, e=e, device=device)

                torch.save(model.state_dict(), f'models/{model_name}_nobias_{dataset_name}_{master_seed}_{i}.pt')
