import torch
from train_helper import evaluate_model
import copy


def fuse_argmax(data_loader: torch.utils.data.DataLoader,
                models: list[torch.nn.Module],
                device: torch.device) \
        -> torch.nn.Module:
    '''
    Selects the best model from "models" by evaluating them with the dataloader.
    '''
    best_acc = 0.0
    best_model = None

    for model in models:
        acc = evaluate_model(model, data_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)

    return best_model
