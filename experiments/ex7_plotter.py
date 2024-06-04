import torch
from matplotlib import pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if __name__ == '__main__':

    model_width = sys.argv[1]  # standard 512 (32,64,128,256,512,1024, 2048)
    model_depth = sys.argv[2]  # standard 4  (1, 2, 4, 8, 16)
    num_models = int(sys.argv[3])  # standard 2 , 4, 8, 16
    master_seed = int(sys.argv[4])

    acc_ensemble = torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_full_ensemble_acc.pt")
    best_individual_acc = torch.load(f'out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_best_individual_acc.pt')
    nt_accuracies = torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_acc.pt")

    iter_and_hierarchical_done = True
    try:
        nt_iterative_accuracies = torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_iterative_acc.pt")
        nt_hierarchical_accuracies = torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_hierarchical_acc.pt")
    except FileNotFoundError:
        iter_and_hierarchical_done = False

    # plot methods under finetuning
    plt.plot(nt_accuracies, label="nt", marker=".", color="y")
    if iter_and_hierarchical_done:
        plt.plot(nt_iterative_accuracies, label="nt_it", marker=".", color="r")
        plt.plot(nt_hierarchical_accuracies, label="nt_h", marker=".", color="b")
    plt.axhline(y=best_individual_acc, color='r', linestyle=':', label="best individual acc")
    plt.axhline(y=acc_ensemble, color='g', linestyle='--', label="output averaging of full ensemble")
    plt.legend()
    plt.title(f"width:{model_width}_depth:{model_depth}_num_models:{num_models}")
    plt.xlabel("finetuning epochs")
    plt.ylabel("accuracy")
    plt.savefig(f"plots/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_.svg", format="svg")

    # print best accuracies of each
    print(f"results for width:{model_width}_depth:{model_depth}_num_models:{num_models}")
    print("ensemble acc: ", acc_ensemble)
    print("best individual: ", best_individual_acc)
    print("nt: ", max(nt_accuracies))
    if iter_and_hierarchical_done:
        print("nt_it; ", max(nt_iterative_accuracies))
        print("nt_h; ", max(nt_hierarchical_accuracies))

