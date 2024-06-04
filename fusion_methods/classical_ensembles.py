import torch


def evaluate_accuracy_output_averaging(data_loader, models, device):
    '''
    Evaluates the ensemble accuracy (output averaging) using the given models and data.
    '''
    # average the outputs

    #  Compute Loss
    correct_pred, num_examples = 0, 0

    for i, (inputs, labels) in enumerate(data_loader):  # Loop over batches in data.
        inputs = inputs.to(device)
        labels = labels.to(device)

        predictions = []  # shape (model_num, batchsize, classes)
        for model in models:
            model.to(device)
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                output = model(inputs)
            predictions.append(output)
        aggregated_predictions = torch.stack(predictions).mean(dim=0)  # (calc mean --> shape (batchsize,classes) )

        _, predicted = torch.max(aggregated_predictions,
                                 dim=1)  # Determine class with max. probability for each sample.
        num_examples += labels.size(0)  # Update overall number of considered samples.
        correct_pred += (predicted == labels).sum()  # Update overall number of correct predictions.

    accuracy = (correct_pred.float() / num_examples).item()
    return accuracy


#
def evaluate_accuracy_majority_vote(data_loader, models, device):
    # do a majority vote
    pass
