import torch
from torch import nn
from train_helper import evaluate_model


def train_distill_ensemble(student: torch.nn.Module,
                           models: list[torch.nn.Module],
                           train_loader: torch.utils.data.DataLoader,
                           valid_loader: torch.utils.data.DataLoader,
                           optimizer: torch.optim.Optimizer,
                           criterion: torch.nn.Module,
                           scheduler: torch.optim.lr_scheduler.LRScheduler,
                           e: int,
                           device: torch.device,
                           T: float = 2,
                           soft_target_loss_weight: float = 0.25) -> list[float]:
    '''
    Trains a given student model with an ensemble teacher using output averaging and knowledge distillation.
    See https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html .

        Parameters:
            student (model): The student model to be trained
            models (array of models): The ensemble teacher
            train_loader: Samples used for the distillation process
            valid_loader: Samples used only for evaluation, not for the training process
            optimizer: Pytorch optimizer
            criterion: Pytorch criterion
            scheduler: Pytorch scheduler
            e: Number of epochs
            device: device
            T: Distillation Temperature
            soft_target_loss_weight: Weight of soft targets. The soft target weight and hard label weight add up to 1.
    '''
    valid_acc_list = []
    student.to(device)

    for model in models:
        model.eval()

    for epoch in range(e):
        student.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            predictions = []
            for model in models:
                model.to(device)
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    output = model(inputs)
                predictions.append(output)

            teacher_logits = torch.stack(predictions).mean(dim=0)
            student_logits = student(inputs)

            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # T is the temperature
            soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T ** 2)

            # criterion = loss_fn = torch.nn.CrossEntropyLoss()
            label_loss = criterion(student_logits, labels)
            loss = soft_target_loss_weight * soft_targets_loss + (1 - soft_target_loss_weight) * label_loss
            loss.backward()
            optimizer.step()

        student.eval()

        # calculate validation accuracy
        _, valid_accuracy = evaluate_model(student, valid_loader, device)
        valid_acc_list.append(valid_accuracy)
        print(f"valid acc in epoch {epoch + 1}:", valid_accuracy)

        scheduler.step()

    return valid_acc_list
