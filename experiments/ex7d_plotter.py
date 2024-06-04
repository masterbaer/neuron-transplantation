import torch
from matplotlib import pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

if __name__ == '__main__':

    model_width = 512  # standard 512 (32,64,128,256,512,1024, 2048)
    model_depth = 4  # standard 4  (1, 2, 4, 8, 16)
    num_models = [2, 4, 8, 16]
    sparsities = [0.0, 0.25, 0.50, 0.75, 0.875, 0.9375, 0.95, 0.99]
    colors = ["#CC0000", "#006600", "#3333FF", "#CC6600"]
    master_seed = 0

    acc_ensemble = torch.load(f"out/ex7d_{model_width}_{model_depth}_16_{master_seed}_full_ensemble_acc.pt")
    best_individual_acc = torch.load(
        f'out/ex7d_{model_width}_{model_depth}_16_{master_seed}_best_individual_acc.pt')

    for (model_count, color) in zip(num_models, colors):
        # create the plot with X: sparsities, Y: best accuracy in finetuning
        sparsity_accuracy_list = []

        for sparsity in sparsities:
            sparsity_accuracy_list.append(max(torch.load(f"out/ex7d_{model_count}_{sparsity}_{master_seed}_nt_acc.pt")))

        plt.plot(sparsities, sparsity_accuracy_list, label=f"{model_count} Models", marker=".", color=color)

        fusion_sparsity = 1 - 1/model_count
        fusion_y_value = None
        for s,v in zip(sparsities, sparsity_accuracy_list):
            if s == fusion_sparsity:
                fusion_y_value = v

        plt.plot([1 - 1 / model_count], [fusion_y_value], marker="x", color=color)

    plt.axhline(y=best_individual_acc, color='r', linestyle=':', label="Best Individual Model")
    plt.axhline(y=acc_ensemble, color='g', linestyle='--', label="Output Averaging of Full Ensemble")

    # plot methods under finetuning

    plt.legend()
    #plt.title(f"accuracies for different sparsities")
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.savefig(f"plots/ex7d.svg", format="svg")
