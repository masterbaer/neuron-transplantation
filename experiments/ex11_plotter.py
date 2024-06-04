import torch
from matplotlib import pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if __name__ == '__main__':

    dataset = sys.argv[1]
    model = sys.argv[2]
    seed = int(sys.argv[3])

    acc_ensemble = torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_full_ensemble_acc.pt")
    best_individual_acc = torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_argmax_acc.pt")
    avg_ft_accuracies = torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_avg_ft_acc.pt")
    nt_ft_accuracies = torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_nt_ft_acc.pt")
    ot_ft_accuracies = torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_ot_ft_acc.pt")
    avg_distill_accuracies = torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_avg_distill_acc.pt")
    model0_distill_accuracies = torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_model0_distill_acc.pt")
    nt_distill_accuracies = torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_nt_distill_acc.pt")
    ot_distill_accuracies = torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_ot_distill_acc.pt")


    # plot methods under finetuning
    plt.plot(avg_ft_accuracies, label="avg", marker=".", color="b")
    plt.plot(nt_ft_accuracies, label="nt", marker=".", color="y")
    plt.plot(ot_ft_accuracies, label="ot", marker=".", color="g")
    plt.axhline(y=best_individual_acc, color='r', linestyle=':', label="best individual acc")
    plt.axhline(y=acc_ensemble, color='g', linestyle='--', label="output averaging of full ensemble")
    plt.legend()
    plt.xlabel("finetuning epochs")
    plt.ylabel("accuracy")
    plt.savefig(f"plots/ex11_{model}_nobias_{dataset}_{seed}_finetuning.svg", format="svg")


    # plot methods under distillation
    plt.cla()
    plt.plot(model0_distill_accuracies, label="distill to model0", marker=".")
    plt.plot(avg_distill_accuracies, label="distill to avg", marker=".", color="b")
    plt.plot(nt_distill_accuracies, label="distill to nt", marker=".", color="y")
    plt.plot(ot_distill_accuracies, label="distill to ot", marker=".", color="g")
    plt.axhline(y=best_individual_acc, color='r', linestyle=':', label="best individual acc")
    plt.axhline(y=acc_ensemble, color='g', linestyle='--', label="output averaging of full ensemble")
    plt.legend()
    plt.xlabel("training epochs")
    plt.ylabel("accuracy")
    plt.savefig(f"plots/ex11_{model}_nobias_{dataset}_{seed}_distillation.svg", format="svg")

    # print best accuracies of each
    print("ensemble acc: ", acc_ensemble)
    print("best individual: ", best_individual_acc)
    print("avg ft: ", max(avg_ft_accuracies))
    print("nt_ft: ", max(nt_ft_accuracies))
    print("ot_ft: ", max(ot_ft_accuracies))
    print("avg_distill: ", max(avg_distill_accuracies))
    print("model0_distill: ", max(model0_distill_accuracies))
    print("nt_distill: ", max(nt_distill_accuracies))
    print("ot_distill: ", max(ot_distill_accuracies))

