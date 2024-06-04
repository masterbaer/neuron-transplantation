from train_helper import evaluate_model

'''
Training procedure using pytorch's training loop. Optionally, a checkpoint with the best validation accuracy is saved.
Adapted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html .
'''


def train_model(device, train_loader, valid_loader, model, optimizer, criterion, scheduler, e, checkpoint=False,
                checkpoint_model=None):
    valid_acc_list = []

    # for checkpoint
    best_valid_acc = 0.0
    if checkpoint:
        checkpoint_model.to(device)

    model.to(device)
    for epoch in range(e):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # criterion = loss_fn = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()

        # calculate validation accuracy
        _, valid_accuracy = evaluate_model(model, valid_loader, device)
        valid_acc_list.append(valid_accuracy)
        print(f"valid acc in epoch {epoch + 1}:", valid_accuracy)
        if checkpoint:
            if valid_accuracy > best_valid_acc:
                best_valid_acc = valid_accuracy
                checkpoint_model.load_state_dict(model.state_dict())
        scheduler.step()

    return valid_acc_list
