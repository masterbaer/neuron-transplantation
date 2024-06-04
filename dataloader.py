import numpy as np
import torch
import torchvision

'''
Dataloaders. We use the image datasets Cifar10, Cifar100, MNIST and SVHN 
(like Entezari et al. https://arxiv.org/pdf/2110.06296.pdf, but omit the imagenet dataset). 
'''

# SET ALL SEEDS

def get_dataloader_from_name(name, batch_size, num_workers=0, root="data", validation_fraction=0.1, model_name=None):
    '''
    Downloads the datasets given a name to the "data" folder and returns train,valid and test data loaders.
    '''
    train_dataset, valid_dataset, test_dataset = None, None, None

    if name == "cifar10":
        # analogous to https://github.com/sidak/otfusion/blob/master/data.py

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # see https: // github.com / soapisnotfat / pytorch - cifar10 / blob / master / main.py for alternative

        # or this:
        # transforms = torchvision.transforms.Compose([
        #             torchvision.transforms.Resize((70, 70)),
        #             torchvision.transforms.RandomCrop((64, 64)),
        #             torchvision.transforms.ToTensor(),
        #             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #         ])

        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transforms, download=True)
        valid_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transforms, download=True)

    if name == "cifar100":
        # from https://github.com/solangii/CIFAR10-CIFAR100/blob/master/data.py
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        train_dataset = torchvision.datasets.CIFAR100(root=root, train=True, transform=transform_train, download=True)
        valid_dataset = torchvision.datasets.CIFAR100(root=root, train=True, transform=transform_train, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, transform=transform_test, download=True)


    if name == "mnist":
        # see https://github.com/sidak/otfusion/blob/master/mnist.py (ot-fusion paper)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=transforms, download=True)
        valid_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.MNIST(root=root, train=False, transform=transforms, download=True)

    if name == "svhn":
        # analogue to here: https://jovian.com/proprincekush/svhn-cnn
        # todo: look at papers to see what they used here
        # todo: get format 1 (not mnist-style but with bounding boxes)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.SVHN(root=root, split="train", transform=transforms, download=True)
        valid_dataset = torchvision.datasets.SVHN(root=root, split="train", transform=transforms, download=True)
        test_dataset = torchvision.datasets.SVHN(root=root, split="test", transform=transforms, download=True)

    if name == "places365":
        pass

    if name == "places205":
        pass

    # Perform index-based train-validation split of original training data.
    total = len(train_dataset)  # Get overall number of samples in original training data.
    idx = list(range(total))  # Make index list.
    np.random.shuffle(idx)  # Shuffle indices.
    vnum = int(validation_fraction * total)  # Determine number of validation samples from validation split.
    train_indices, valid_indices = idx[vnum:], idx[0:vnum]  # Extract train and validation indices.

    # Get samplers.
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    # Get data loaders.
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               num_workers=num_workers, sampler=valid_sampler)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               num_workers=num_workers, drop_last=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=False)

    return train_loader, valid_loader, test_loader


# GET DATALOADERS (NON-PARALLEL)
def get_dataloaders_cifar10(batch_size,
                            num_workers=0,
                            root='cifar10',
                            validation_fraction=0.1,
                            train_transforms=None,
                            test_transforms=None):
    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()

    if test_transforms is None:
        test_transforms = torchvision.transforms.ToTensor()

    # Load training data.
    train_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=train_transforms,
        download=True
    )

    # Load validation data.
    valid_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=test_transforms
    )

    # Load test data.
    test_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        transform=test_transforms
    )

    # Perform index-based train-validation split of original training data.
    total = len(train_dataset)  # Get overall number of samples in original training data.
    idx = list(range(total))  # Make index list.
    np.random.shuffle(idx)  # Shuffle indices.
    vnum = int(validation_fraction * total)  # Determine number of validation samples from validation split.
    train_indices, valid_indices = idx[vnum:], idx[0:vnum]  # Extract train and validation indices.

    # Get samplers.
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    # Get data loaders.
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=valid_sampler
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        sampler=train_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader