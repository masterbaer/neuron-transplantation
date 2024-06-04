# mean/std for the comparison of ex7c

import torch
import statistics
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    num_models_list = [2, 4, 8, 16]
    seeds = [0, 1, 2, 3, 4]
    model_width = 512
    model_depth = 4

    for num_models in num_models_list:
        acc_ensemble_list = []
        best_individual_acc_list = []

        nt_ft_best_accuracies_list = []
        nt_iterative_ft_best_accuracies_list = []
        nt_hierarchical_ft_best_accuracies_list = []

        nt_ft_starting_accuracies_list = []
        nt_iterative_ft_starting_accuracies_list = []
        nt_hierarchical_ft_starting_accuracies_list = []

        for master_seed in seeds:
            acc_ensemble_list.append(
                torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_full_ensemble_acc.pt"))
            best_individual_acc_list.append(
                torch.load(f'out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_best_individual_acc.pt'))

            nt_ft_best_accuracies_list.append(
                max(torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_acc.pt")[:20]))
            nt_iterative_ft_best_accuracies_list.append(
                max(torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_iterative_acc.pt")[:20]))
            nt_hierarchical_ft_best_accuracies_list.append(
                max(torch.load(
                    f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_hierarchical_acc.pt")[:20]))

            nt_ft_starting_accuracies_list.append(
                torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_acc.pt")[0])
            nt_iterative_ft_starting_accuracies_list.append(
                torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_iterative_acc.pt")[0])
            nt_hierarchical_ft_starting_accuracies_list.append(
                torch.load(f"out/ex7_{model_width}_{model_depth}_{num_models}_{master_seed}_nt_hierarchical_acc.pt")[0])

        print(f"printing stats for {num_models}")

        print("ensemble acc: ", round(100 * statistics.mean(acc_ensemble_list), 2), "+-",
              round(100 * statistics.stdev(acc_ensemble_list), 2))
        print("best individual acc: ", round(100 * statistics.mean(best_individual_acc_list), 2), "+-",
              round(100 * statistics.stdev(best_individual_acc_list), 2))
        print("nt start: ", round(100 * statistics.mean(nt_ft_starting_accuracies_list), 2), "+-",
              round(100 * statistics.stdev(nt_ft_starting_accuracies_list), 2))
        print("nt_it start: ", round(100 * statistics.mean(nt_iterative_ft_starting_accuracies_list), 2), "+-",
              round(100 * statistics.stdev(nt_iterative_ft_starting_accuracies_list), 2))
        print("nt_h start: ", round(100 * statistics.mean(nt_hierarchical_ft_starting_accuracies_list), 2), "+-",
              round(100 * statistics.stdev(nt_hierarchical_ft_starting_accuracies_list), 2))

        print("nt best: ", round(100 * statistics.mean(nt_ft_best_accuracies_list), 2), "+-",
              round(100 * statistics.stdev(nt_ft_best_accuracies_list), 2))
        print("nt_it best: ", round(100 * statistics.mean(nt_iterative_ft_best_accuracies_list), 2), "+-",
              round(100 * statistics.stdev(nt_iterative_ft_best_accuracies_list), 2))
        print("nt_h best: ", round(100 * statistics.mean(nt_hierarchical_ft_best_accuracies_list), 2), "+-",
              round(100 * statistics.stdev(nt_hierarchical_ft_best_accuracies_list), 2))
