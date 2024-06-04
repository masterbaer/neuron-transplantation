# make plot of immediate accuracies and finetuned accuracies!
# plot should contain transfer rates of 0/8, 1/8, 2/8, ... 8/8
import torch
from matplotlib import pyplot as plt

if __name__ == "__main__":

    transfer_rates = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    immediate_accs = []
    best_accs = []

    for rate in transfer_rates:

        immediate_accs.append(torch.load(f'out/ex9_{rate}_nt_acc.pt')[0])
        best_accs.append(max(torch.load(f'out/ex9_{rate}_nt_acc.pt')))

    # plot

    plt.plot(transfer_rates,immediate_accs, label="Without Fine-tuning", marker=".", color="#006600")
    plt.plot(transfer_rates, best_accs, label="With Fine-tuning", marker=".", color="#CC0000")
    plt.legend()

    #plt.title(f"transplanting a different amount of neurons")
    plt.xlabel("Transplantation Percentage")
    plt.ylabel("Accuracy")
    plt.savefig(f"plots/ex9.svg", format="svg")
