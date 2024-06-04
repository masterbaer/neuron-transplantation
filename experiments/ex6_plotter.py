import numpy as np
import torch
from matplotlib import pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


if __name__ == '__main__':

    seeds = [0, 1, 2, 3, 4]

    all_acc_m_p_ft = []  # to be averaged along seed axis to get mean results
    all_acc_p_m_ft = []
    all_acc_m_ft_p_ft = []

    # accuracies consist of
    # m_(ft)_p_ft: merged_acc, (3 finetuned acc), 20 finetune acc
    # p_m_ft: 20 finetune


    for seed in seeds:


        accuracies_m_p_ft = torch.load(f"out/ex6_m_p_ft_{seed}_acc.pt")
        accuracies_m_p_ft = accuracies_m_p_ft[1:]
        all_acc_m_p_ft.append(accuracies_m_p_ft)

        accuracies_p_m_ft = torch.load(f"out/ex6_p_m_ft_{seed}_acc.pt")
        all_acc_p_m_ft.append(accuracies_p_m_ft)

        accuracies_m_ft_p_ft = torch.load(f"out/ex6_m_ft_p_ft_{seed}_acc.pt")
        accuracies_m_ft_p_ft = accuracies_m_ft_p_ft[4:-3] # remove initial finetuning epochs and last 3 (as it is too long)
        all_acc_m_ft_p_ft.append(accuracies_m_ft_p_ft)

        plt.cla()
        num_epochs = len(accuracies_p_m_ft)
        x = np.linspace(0, num_epochs, num_epochs)  # epochs
        x_without_ft = np.linspace(3, num_epochs, num_epochs-3)  # epochs

        plt.plot(x, accuracies_m_p_ft, label="m_p_ft", marker=".", color="b")
        plt.plot(x, accuracies_p_m_ft, label="p_m_ft", marker=".", color="y")
        plt.plot(x_without_ft, accuracies_m_ft_p_ft, label="m_ft_p_ft", marker=".", color="g")

        plt.legend()
        plt.xlabel("finetuning epochs")
        plt.ylabel("accuracy")

        plt.savefig(f"plots/ex6_{seed}.svg", format="svg")

    # mean plots:

    num_epochs = len(all_acc_p_m_ft[0])
    x = np.linspace(0, num_epochs, num_epochs)  # epochs
    all_acc_m_p_ft = np.array(all_acc_m_p_ft)
    all_acc_p_m_ft = np.array(all_acc_p_m_ft)
    all_acc_m_ft_p_ft = np.array(all_acc_m_ft_p_ft)
    x_without_ft = np.linspace(3, num_epochs, num_epochs-3)  # epochs

    y_m_p_ft = np.mean(all_acc_m_p_ft, axis=0)
    y_p_m_ft = np.mean(all_acc_p_m_ft, axis=0)
    y_m_ft_p_ft = np.mean(all_acc_m_ft_p_ft, axis=0)

    plt.cla()
    plt.plot(x, y_m_p_ft, label="m_p_ft", marker=".", color="b")
    plt.plot(x, y_p_m_ft, label="p_m_ft", marker=".", color="y")
    plt.plot(x_without_ft, y_m_ft_p_ft, label="m_ft_p_ft", marker=".", color="g")
    #plt.plot(y_m_ft_p_ft[0:3], marker="x", color="g")

    plt.title("mean curves")
    plt.legend()
    plt.xlabel("epochs/events")
    plt.ylabel("accuracy")
    plt.savefig(f"plots/ex6_mean.svg", format="svg")

    for e in range(num_epochs):
        print("epoch", e)
        print("p_m_ft mean after epoch ", e, ": ", y_p_m_ft[e])
        print("m_p_ft mean after epoch ", e, ": ", y_m_p_ft[e])
        if e >= 3:
            print("m_ft_p_ft mean after epoch ", e, ": ", y_m_ft_p_ft[e-3])
