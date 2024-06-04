'''
Here we compare the time and memory requirements of Vanilla Averaging, NT and OT.
'''

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataloader import get_dataloader_from_name
from model import AdaptiveNeuralNetwork2
from fusion_methods.neuron_transplantation import fuse_ensemble
from fusion_methods.optimal_transport import fuse_optimal_transport
from fusion_methods.weight_averaging import average_weights

if __name__ == "__main__":
    # testing one layer fusion with different widths
    dataset_name = "cifar10"  # just use cifar10
    method = sys.argv[1]  # avg, ot, nt
    layer_width = int(sys.argv[2])  # 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384...
    print("method: ", method, "width: ", layer_width)
    num_classes = 10
    num_models = 2

    b = 256
    train_loader, valid_loader, test_loader = get_dataloader_from_name(dataset_name, root=dataset_name, batch_size=b)
    image_shape = None
    example_inputs = None
    for images, labels in train_loader:
        image_shape = images.shape
        example_inputs = images
        break
    input_dim = image_shape[1] * image_shape[2] * image_shape[3]

    models = []
    for i in range(num_models):
        model = AdaptiveNeuralNetwork2(input_dim, num_classes, layer_width=layer_width, num_layers=2)
        models.append(model)

    # todo measure time and ram. doing iterative version of NT is possible too
    start = time.time()

    if method == "avg":
        fused_model = average_weights(models)

    if method == "nt":
        fused_model = fuse_ensemble(models, example_inputs)

    if method == "ot":
        fused_model = fuse_optimal_transport(models)
        # adaptive 6 not possible due to RAM?

    end = time.time()
    print("time elapsed: ", end - start)

# we have a 38.67GB RAM limit on these experiments

# layer width   32    ,                64,                128,             256,          512,
# method  avg  0.00089s/3.79 MB, .00112s/3.76 MB   0.0015s/3.79 MB   0.0023s/3.75 MB  0.005s/3.82 MB
#         nt   0.00637s/3.75 MB,  0.009s/3.80 MB   0.084s/3.76 MB    0.025s/3.79 MB   0.056s/3.75 MB
#         ot   0.010s/3.78 MB,   0.045/3.76 MB     0.18s/3.77 MB     0.731s/3.76 MB   3.19s/3.82 MB

   # 1024       ,   2048,          4096,              8192,               16384
# 0.015s/3.76 MB 0.044s/3.75 MB    0.119s/3,82 MB  0.373s / 3.74 MB     1.300/3.80 MB
# 0.146s/3.76 MB  0.47s/3.81 MB    1.39s/3.80 MB   4.91s/3.65 GB        18.43s/10.98 GB
# 14.57s/22.14 GB    /              /                /                  /

   # 384               #768                 # 1536
# 0.0034s/3.82MB     # 0.010s/3.75 MB       # 0.032s/3.75 MB
# 0.0424s/3.75MB     # 0.098s / 3.83 MB     # 0.287s/3.85 MB
# 1.773s/3.80 MB   # 8.179s / 13.31 GB      # /



# vit_gigantic_patch14_224 has embed_dim of 1664 in
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

