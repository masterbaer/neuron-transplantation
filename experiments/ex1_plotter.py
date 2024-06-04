import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':

    accuracies_model0 = torch.load("out/ex1_test_accuracy0.pt")
    accuracies_model1 = torch.load("out/ex1_test_accuracy1.pt")
    accuracies_model2 = torch.load("out/ex1_test_accuracy2.pt")
    accuracies_model3 = torch.load("out/ex1_test_accuracy3.pt")

    accuracies_combined = torch.load("out/ex1_combined_test_accuracy.pt")
    accuracies_combined_scratch = torch.load("out/ex1_combined_scratch_test_accuracy.pt")

    plt.plot(accuracies_model0, label="model 0")
    plt.plot(accuracies_model1, label="model 1")
    plt.plot(accuracies_model2, label="model 2")
    plt.plot(accuracies_model3, label="model 3")
    plt.plot(accuracies_combined, label="output averaging")
    plt.plot(accuracies_combined_scratch, label="big model from scratch")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracies")
    plt.savefig("plots/ex1_accuracies.png")
    plt.cla()
