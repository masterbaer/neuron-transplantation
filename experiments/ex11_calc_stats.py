import torch
import statistics
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # models = ["smallnn", "lenet", "resnet18", "vit"]
    # for now only smallnn

    models = ["vgg11"]
    datasets = ["cifar10", "cifar100"]
    seeds = [0, 1, 2, 3, 4]

    # seeds = [0]

    # TODO: report mean/std after 3 epochs and best acc for a) ft and b) distillation for all methods!

    for model in models:
        for dataset in datasets:
            acc_ensemble_list = []
            best_individual_acc_list = []

            avg_starting_acc_list = []
            nt_starting_acc_list = []
            ot_starting_acc_list = []

            avg_ft_accuracies_list = []
            nt_ft_accuracies_list = []
            ot_ft_accuracies_list = []

            model0_distill_accuracies_list = []
            avg_distill_accuracies_list = []
            ot_distill_accuracies_list = []
            nt_distill_accuracies_list = []

            for seed in seeds:
                acc_ensemble_list.append(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_full_ensemble_acc.pt"))
                best_individual_acc_list.append(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_argmax_acc.pt"))

                avg_starting_acc_list.append(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_avg_ft_acc.pt")[0])
                ot_starting_acc_list.append(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_ot_ft_acc.pt")[0])
                nt_starting_acc_list.append(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_nt_ft_acc.pt")[0])

                avg_ft_accuracies_list.append(
                    max(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_avg_ft_acc.pt")))
                ot_ft_accuracies_list.append(max(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_ot_ft_acc.pt")))
                nt_ft_accuracies_list.append(max(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_nt_ft_acc.pt")))

                model0_distill_accuracies_list.append(
                    max(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_avg_distill_acc.pt")))
                avg_distill_accuracies_list.append(
                    max(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_avg_distill_acc.pt")))
                ot_distill_accuracies_list.append(
                    max(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_ot_distill_acc.pt")))
                nt_distill_accuracies_list.append(
                    max(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_nt_distill_acc.pt")))

            print(f"printing stats for {model} and {dataset}")
            print("ensemble acc: ", round(100 * statistics.mean(acc_ensemble_list), 2), "+-",
                  round(100 * statistics.stdev(acc_ensemble_list), 2))
            print("best individual acc: ", round(100 * statistics.mean(best_individual_acc_list), 2), "+-",
                  round(100 * statistics.stdev(best_individual_acc_list), 2))

            print("avg student: ", round(100 * statistics.mean(avg_starting_acc_list), 2), "+-",
                  round(100 * statistics.stdev(avg_starting_acc_list), 2))
            print("ot student: ", round(100 * statistics.mean(ot_starting_acc_list), 2), "+-",
                  round(100 * statistics.stdev(ot_starting_acc_list), 2))
            print("nt student: ", round(100 * statistics.mean(nt_starting_acc_list), 2), "+-",
                  round(100 * statistics.stdev(nt_starting_acc_list), 2))

            print("avg ft: ", round(100 * statistics.mean(avg_ft_accuracies_list), 2), "+-",
                  round(100 * statistics.stdev(avg_ft_accuracies_list), 2))
            print("ot_ft: ", round(100 * statistics.mean(ot_ft_accuracies_list), 2), "+-",
                  round(100 * statistics.stdev(ot_ft_accuracies_list), 2))
            print("nt_ft: ", round(100 * statistics.mean(nt_ft_accuracies_list), 2), "+-",
                  round(100 * statistics.stdev(nt_ft_accuracies_list), 2))

            print("model0_distill: ", round(100 * statistics.mean(model0_distill_accuracies_list), 2), "+-",
                  round(100 * statistics.stdev(model0_distill_accuracies_list), 2))
            print("avg_distill: ", round(100 * statistics.mean(avg_distill_accuracies_list), 2), "+-",
                  round(100 * statistics.stdev(avg_distill_accuracies_list), 2))
            print("ot_distill: ", round(100 * statistics.mean(ot_distill_accuracies_list), 2), "+-",
                  round(100 * statistics.stdev(ot_distill_accuracies_list), 2))
            print("nt_distill: ", round(100 * statistics.mean(nt_distill_accuracies_list), 2), "+-",
                  round(100 * statistics.stdev(nt_distill_accuracies_list), 2))

            print("----------------------------------------------------")

            # make mean/stdv plot, see https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars

            all_ot_ft_accuracies_list = []
            all_nt_ft_accuracies_list = []
            all_avg_ft_accuracies_list = []

            all_ot_distill_accuracies_list = []
            all_nt_distill_accuracies_list = []
            all_avg_distill_accuracies_list = []
            all_model0_distill_accuracies_list = []

            for seed in seeds:
                all_ot_ft_accuracies_list.append(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_ot_ft_acc.pt"))
                all_nt_ft_accuracies_list.append(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_nt_ft_acc.pt"))
                all_avg_ft_accuracies_list.append(torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_avg_ft_acc.pt"))
                all_ot_distill_accuracies_list.append(
                    torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_ot_distill_acc.pt"))
                all_nt_distill_accuracies_list.append(
                    torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_nt_distill_acc.pt"))
                all_avg_distill_accuracies_list.append(
                    torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_avg_distill_acc.pt"))
                all_model0_distill_accuracies_list.append(
                    torch.load(f"out/ex11_{model}_nobias_{dataset}_{seed}_model0_distill_acc.pt"))
            print("-------")
            for i in range(10 + 1):
                accs_avg_ft_after_i = [accs[i] for accs in all_avg_ft_accuracies_list]
                accs_ot_ft_after_i = [accs[i] for accs in all_ot_ft_accuracies_list]
                accs_nt_ft_after_i = [accs[i] for accs in all_nt_ft_accuracies_list]

                accs_model0_distill_after_i = [accs[i] for accs in all_model0_distill_accuracies_list]
                accs_avg_distill_after_i = [accs[i] for accs in all_avg_distill_accuracies_list]
                accs_ot_distill_after_i = [accs[i] for accs in all_ot_distill_accuracies_list]
                accs_nt_distill_after_i = [accs[i] for accs in all_nt_distill_accuracies_list]



                print(f"ot_ft after {i}: ", round(100 * statistics.mean(accs_ot_ft_after_i), 2), "+-",
                      round(100 * statistics.stdev(accs_ot_ft_after_i), 2))
                print(f"nt_ft after {i}: ", round(100 * statistics.mean(accs_nt_ft_after_i), 2), "+-",
                      round(100 * statistics.stdev(accs_nt_ft_after_i), 2))
                print(f"avg_ft after {i}: ", round(100 * statistics.mean(accs_avg_ft_after_i), 2), "+-",
                      round(100 * statistics.stdev(accs_avg_ft_after_i), 2))
                print(f"ot_distill after {i}: ", round(100 * statistics.mean(accs_ot_distill_after_i), 2), "+-",
                      round(100 * statistics.stdev(accs_ot_distill_after_i), 2))
                print(f"nt_distill after {i}: ", round(100 * statistics.mean(accs_nt_distill_after_i), 2), "+-",
                      round(100 * statistics.stdev(accs_nt_distill_after_i), 2))
                print(f"avg_distill after {i}: ", round(100 * statistics.mean(accs_avg_distill_after_i), 2), "+-",
                      round(100 * statistics.stdev(accs_avg_distill_after_i), 2))
                print(f"model0_distill after {i}: ", round(100 * statistics.mean(accs_model0_distill_after_i), 2), "+-",
                      round(100 * statistics.stdev(accs_model0_distill_after_i), 2))

                print("---")

            all_ot_ft_accuracies_list = np.array(all_ot_ft_accuracies_list)
            all_nt_ft_accuracies_list = np.array(all_nt_ft_accuracies_list)
            all_avg_ft_accuracies_list = np.array(all_avg_ft_accuracies_list)
            all_ot_distill_accuracies_list = np.array(all_ot_distill_accuracies_list)
            all_nt_distill_accuracies_list = np.array(all_nt_distill_accuracies_list)
            all_avg_distill_accuracies_list = np.array(all_avg_distill_accuracies_list)
            all_model0_distill_accuracies_list = np.array(all_model0_distill_accuracies_list)

            best_individual_acc = statistics.mean(best_individual_acc_list)
            acc_ensemble = statistics.mean(acc_ensemble_list)

            num_epochs = len(all_ot_ft_accuracies_list[0, :])
            x = np.linspace(0, num_epochs, num_epochs)  # epochs

            y_ot_ft = np.mean(all_ot_ft_accuracies_list, axis=0)
            err_ot_ft = np.std(all_ot_ft_accuracies_list, axis=0)

            y_nt_ft = np.mean(all_nt_ft_accuracies_list, axis=0)
            err_nt_ft = np.std(all_nt_ft_accuracies_list, axis=0)

            y_avg_ft = np.mean(all_avg_ft_accuracies_list, axis=0)
            err_avg_ft = np.std(all_avg_ft_accuracies_list, axis=0)

            y_ot_distill = np.mean(all_ot_distill_accuracies_list, axis=0)
            err_ot_distill = np.std(all_ot_distill_accuracies_list, axis=0)

            y_nt_distill = np.mean(all_nt_distill_accuracies_list, axis=0)
            err_nt_distill = np.std(all_nt_distill_accuracies_list, axis=0)

            y_avg_distill = np.mean(all_avg_distill_accuracies_list, axis=0)
            err_avg_distill = np.std(all_avg_distill_accuracies_list, axis=0)

            plt.cla()
            plt.plot(x, y_ot_ft, marker=None, color="#006600", linewidth=0.5, label="OT")
            plt.plot(x, y_nt_ft, marker=None, color="#CC0000", linewidth=0.5, label="NT")
            plt.plot(x, y_avg_ft, marker=None, color="#3333FF", linewidth=0.5, label="Avg")
            plt.xlabel("Fine-tuning Epochs")
            plt.ylabel("Accuracy")
            #plt.title(f"Recovering accuracy using finetuning on {model}/{dataset}")
            plt.axhline(y=best_individual_acc, color='#CC0000', linestyle=':', label="Best Individual Model")
            plt.axhline(y=acc_ensemble, color='#006600', linestyle='--', label="Output Averaging of Full Ensemble")
            plt.fill_between(x, y_ot_ft - err_ot_ft, y_ot_ft + err_ot_ft, color="#D5E8D4", alpha=0.5)
            plt.fill_between(x, y_nt_ft - err_nt_ft, y_nt_ft + err_nt_ft, color="#F8CECC", alpha=0.5)
            plt.fill_between(x, y_avg_ft - err_avg_ft, y_avg_ft + err_avg_ft, color="#DAE8FC", alpha=0.5)
            plt.legend()
            plt.savefig(f"plots/ex11_mean_std_{model}_{dataset}_ft.svg", format="svg")

            plt.cla()
            plt.plot(x, y_ot_distill, marker=None, color="#006600", linewidth=0.5, label="OT")
            plt.plot(x, y_nt_distill, marker=None, color="#CC0000", linewidth=0.5, label="NT")
            plt.plot(x, y_avg_distill, marker=None, color="#3333FF", linewidth=0.5, label="Avg")
            plt.xlabel("Distillation Epochs")
            plt.ylabel("Accuracy")
            #plt.title(f"Recovering accuracy using distillation on {model}/{dataset}")
            plt.axhline(y=best_individual_acc, color='#CC0000', linestyle=':', label="Best Individual Model")
            plt.axhline(y=acc_ensemble, color='#006600', linestyle='--', label="Output Averaging of Full Ensemble")
            plt.fill_between(x, y_ot_distill - err_ot_distill, y_ot_distill + err_ot_distill, color="#D5E8D4", alpha=0.5)
            plt.fill_between(x, y_nt_distill - err_nt_distill, y_nt_distill + err_nt_distill, color="#F8CECC", alpha=0.5)
            plt.fill_between(x, y_avg_distill - err_avg_distill, y_avg_distill + err_avg_distill, color="#DAE8FC", alpha=0.5)
            plt.legend()
            plt.savefig(f"plots/ex11_mean_std_{model}_{dataset}_distill.svg", format="svg")
