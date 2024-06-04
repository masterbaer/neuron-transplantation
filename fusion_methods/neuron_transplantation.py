import copy

import torch
from torch import nn
import torch_pruning as tp
from functools import reduce
from typing import Union

'''
Implementation of Neuron Transplantation and its variants as a fusion method. 
'''


def fuse_ensemble(models, example_inputs):
    '''
    Neuron Transplantation. The models are fused by taking their most important neurons.
    First, a large, vertically concatenated model is created where all ensemble members are represented.
    Then the large model is pruned, selecting the largest neurons across all models.
    Note that this method requires the models to be on CPU since torch-pruning does so as well.

    Parameters:
        models: Ensemble members to be fused
        example_inputs: Input data (one sample is enough) to feed into torch-pruning.

    Return: NT-fused model of the original architecture.
    '''
    # assumes all inputs to be on cpu

    combined_model = concat_models(models)
    num_models = len(models)
    sparsity = 1 - (1 / num_models)

    last_layer_name = getLastLayerName(combined_model)
    last_layer = get_module_by_name(combined_model, last_layer_name)

    # bring everything to cpu for the combination (for torch-pruning)

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        combined_model,
        example_inputs,
        importance=imp,
        pruning_ratio=sparsity,
        root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
        ignored_layers=[last_layer],
    )
    pruner.step()

    return combined_model


def fuse_ensemble_iterative(models, example_inputs):
    '''
    Iterative version of Neuron Transplantation. The models are iteratively fused in a "running average"-manner.
    Hence, the first models have little weight and the last models have a lot of weight.
    '''
    fused_model = models[0]

    for model in models[1:]:
        # fuse the currently fused model with the new model

        # combine
        fused_model = concat_models([fused_model, model])

        # prune
        last_layer_name = getLastLayerName(fused_model)
        last_layer = get_module_by_name(fused_model, last_layer_name)
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner = tp.pruner.MagnitudePruner(
            fused_model,
            example_inputs,
            importance=imp,
            pruning_ratio=0.5,
            root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
            ignored_layers=[last_layer],
        )
        pruner.step()

    return fused_model


def fuse_ensemble_hierarchical(models, example_inputs):
    '''
    Hierarchical version of Neuron Transplantation. The models are fused recursively in a merge-sort-manner.
    '''
    # similar to mergesort, just with fusions
    num_models = len(models)

    if num_models >= 4:
        midpoint = num_models // 2
        left = fuse_ensemble_hierarchical(models[:midpoint], example_inputs)
        right = fuse_ensemble_hierarchical(models[midpoint:], example_inputs)
        return fuse_ensemble([left, right], example_inputs)
    else:
        # 2 or 3 models , this is the base case
        return fuse_ensemble(models, example_inputs)


def concat_models(models):
    '''
    Create a large model with vertically concatenated layers. Each layer is iterated through simultaneously across all
    models. When a certain layer type is reached, e.g. Linear/Conv2D/BatchNorm2D, a large layer is created and
    all individual layers are copied to the large layer.
    '''
    input_layer_name = getFirstLayerName(models[0])
    output_layer_name = getLastLayerName(models[0])

    # use the first model as the target to create the combined model
    combined_model = copy.deepcopy(models[0])

    # go through each layer of every model at the same time
    iters = [iter(model.named_modules()) for model in models]
    iters.append(iter(combined_model.named_modules()))
    # list [(layername,layer)] of named modules

    # make a list of layers that are to be changed in the target model and assign the layers at the end
    updated_layers = []

    while True:
        is_first_layer = False
        is_last_layer = False

        # get next leaf layers
        try:
            layer_list = [next(iter) for iter in iters]
        except StopIteration:
            break

        while not (isLayerWithParams(layer_list[0][1])):
            try:
                layer_list = [next(iter) for iter in iters]
            except StopIteration:
                break

        layer_name = None
        for module in layer_list:
            name, layer = module
            layer_name = name
            break

        if layer_name == input_layer_name:
            is_first_layer = True
        if layer_name == output_layer_name:
            is_last_layer = True

        # currently at a leaf layer, i.e. at a Linear layer / Conv2D layer / BatchNorm2d layer
        # merge layers

        if isinstance(layer_list[0][1], torch.nn.Linear):
            worker_list = [layer for (name, layer) in layer_list[:-1]]
            large_layer = combine_linear_layers(worker_list, is_first_layer, is_last_layer)

            # write data to target layer
            target_name, target_layer = layer_list[-1]
            updated_layers.append((target_name, large_layer))

        if isinstance(layer_list[0][1], torch.nn.Conv2d):
            worker_list = [layer for (name, layer) in layer_list[:-1]]
            large_layer = combine_conv_layers(worker_list, is_first_layer)

            # write data to target layer
            target_name, target_layer = layer_list[-1]

            updated_layers.append((target_name, large_layer))

        if isinstance(layer_list[0][1], torch.nn.BatchNorm2d):
            worker_list = [layer for (name, layer) in layer_list[:-1]]
            large_layer = combine_batchnorm_layers(worker_list)

            target_name, target_layer = layer_list[-1]
            updated_layers.append((target_name, large_layer))

    # assign all layers to the combined model
    for target_name, large_layer in updated_layers:
        set_module(combined_model, target_name, large_layer)
        # setattr(combined_model, target_name, large_layer) does not work as it is not recursive

    return combined_model


def combine_linear_layers(worker_layers, is_first_layer, is_last_layer):
    '''
    Vertical Concatenation of linear layers. Depending on the position (first layer, middle layer, last layer),
    the layers are concatenated with similar inputs (first layer), different inputs (not first layer) and either
    averaged (output layer) or not averaged (not last layer).

    Parameters:
        worker_layers: List of layers to be concatenated.
        is_first_layer (bool): If the layers are the first layers each, then they use the same input. Otherwise not.
        is_last_layer (bool): If the layers act as the classification layers, then the outputs are to be averaged.

    Return: A vertically concatenated linear layer.
    '''
    num_out = 0
    num_in = 0
    bias = worker_layers[0].bias is not None

    for layer in worker_layers:
        num_in += layer.in_features  # layer.weight.shape[1]
        num_out += layer.out_features  # layer.weight.shape[0]

    # the first and last layers are special as they all use the same input/output respectively
    if is_first_layer:
        for layer in worker_layers:
            # overwrite
            num_in = layer.in_features  # or use layer.weight.shape[1]
            break

    if is_last_layer:
        for layer in worker_layers:
            # overwrite
            num_out = layer.out_features  # or use layer.weight.shape[1]
            break

    # print(num_in, num_out)

    # create a large linear layer with zeros
    large_layer = nn.Linear(out_features=num_out, in_features=num_in, bias=bias)
    large_layer.weight.data.fill_(0.0)
    if bias:
        large_layer.bias.data.fill_(0.0)

    if is_first_layer:
        output_index = 0
        for layer in worker_layers:
            current_out_size = layer.out_features

            large_layer.weight.data[output_index:output_index + current_out_size, :] += layer.weight.data
            if bias:
                large_layer.bias.data[output_index:output_index + current_out_size] += layer.bias.data

            # increase index
            output_index += current_out_size

    elif is_last_layer:
        input_index = 0

        # calculate the total in_size for weighting (or use 1/p for similar architectures)
        total_in_size = 0
        for layer in worker_layers:
            total_in_size += layer.in_features

        for layer in worker_layers:
            current_in_size = layer.in_features
            scaling = current_in_size / total_in_size

            large_layer.weight.data[:, input_index:input_index + current_in_size] += layer.weight.data * scaling
            if bias:
                large_layer.bias.data[:] += layer.bias.data * scaling

            # increase index
            input_index += current_in_size

    else:
        input_index = 0
        output_index = 0

        for layer in worker_layers:
            current_in_size = layer.in_features
            current_out_size = layer.out_features

            large_layer.weight.data[output_index:output_index + current_out_size,
            input_index:input_index + current_in_size] += layer.weight.data
            if bias:
                large_layer.bias.data[output_index:output_index + current_out_size] += layer.bias.data

            # increase index
            input_index += current_in_size
            output_index += current_out_size

    return large_layer


def combine_conv_layers(worker_layers, is_first_layer):
    '''
    Channel-wise concatenation of convolutional layers.

    Parameters:
        worker_layers: layers to be concatenated
        is_first_layer (bool): status to decide whether the inputs are similar (for the first layer) or different.

    Return: Concatenated convolutional layer.
    '''

    kernel_size = worker_layers[0].kernel_size
    stride = worker_layers[0].stride
    padding = worker_layers[0].padding
    dilation = worker_layers[0].dilation
    bias = worker_layers[0].bias is not None
    groups = worker_layers[0].groups
    padding_mode = worker_layers[0].padding_mode

    # calculate target shape depending on layer position (first or last or else)
    num_out = 0
    num_in = 0

    for layer in worker_layers:
        num_in += layer.in_channels  # layer.weight.shape[1]
        num_out += layer.out_channels  # layer.weight.shape[0]

    # the first and last layers are special as they all use the same input/output respectively
    if is_first_layer:
        for layer in worker_layers:
            # overwrite
            num_in = layer.in_channels  # or use layer.weight.shape[1]
            break

    # create a large convoluional layer with zeros
    large_layer = nn.Conv2d(in_channels=num_in, out_channels=num_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias,
                            groups=groups, padding_mode=padding_mode, dilation=dilation)
    large_layer.weight.data.fill_(0.0)

    if bias:
        large_layer.bias.data.fill_(0.0)

    # fill the layer
    if is_first_layer:
        input_index = 0
        output_index = 0
        for layer in worker_layers:
            current_out_size = layer.out_channels

            large_layer.weight.data[output_index:output_index + current_out_size, :, :, :] += layer.weight.data
            if bias:
                large_layer.bias.data[output_index:output_index + current_out_size] += layer.bias.data

            # increase index
            output_index += current_out_size
    else:
        # layer somewhere in the middle
        input_index = 0
        output_index = 0

        for layer in worker_layers:
            current_in_size = layer.in_channels
            current_out_size = layer.out_channels

            large_layer.weight.data[output_index:output_index + current_out_size,
            input_index:input_index + current_in_size, :, :] += layer.weight.data

            if bias:
                large_layer.bias.data[output_index:output_index + current_out_size] += layer.bias.data

            # increase index
            input_index += current_in_size
            output_index += current_out_size

    return large_layer


def combine_batchnorm_layers(worker_modules):
    '''
    Concatenation of batch-norm layers. All parameters are stacked.
    '''
    eps = worker_modules[0].eps
    momentum = worker_modules[0].momentum
    affine = worker_modules[0].affine
    track_running_stats = worker_modules[0].track_running_stats
    bias = worker_modules[0].bias is not None

    num_features = 0

    for layer in worker_modules:
        num_features += layer.num_features

    large_layer = nn.BatchNorm2d(num_features=num_features, affine=affine,
                                 track_running_stats=track_running_stats, eps=eps, momentum=momentum)

    large_layer.weight.data.fill_(0.0)
    if bias:
        large_layer.bias.data.fill_(0.0)

    large_layer.weight.data = torch.cat([layer.weight.data for layer in worker_modules])
    if bias:
        large_layer.bias.data = torch.cat([layer.bias.data for layer in worker_modules])
    large_layer.running_mean.data = torch.cat([layer.running_mean.data for layer in worker_modules])
    large_layer.running_var.data = torch.cat([layer.running_var.data for layer in worker_modules])
    return large_layer


def isLeafLayer(layer):
    return len(list(layer.children())) == 0


def isLayerWithParams(layer):
    return isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d) or isinstance(layer,
                                                                                                  torch.nn.BatchNorm2d)


def getFirstLayerName(model):
    first_layer_name = None
    for name, module in model.named_modules():
        if isLayerWithParams(module):
            if first_layer_name is None:
                first_layer_name = name
    return first_layer_name


def getLastLayerName(model):
    last_layer_name = None
    for name, module in model.named_modules():
        if isLayerWithParams(module):
            last_layer_name = name
    return last_layer_name


def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                       access_string: str):
    '''
    Retrieves a module (e.g. a linear layer weight) nested in another by its access string (e.g. layer1.0.weight)
    See https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8
    '''
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def set_module(module, access_string, new_module):
    '''
    Replaces a module with another using the access string.
    '''
    names = access_string.split(sep='.')
    parent_module = reduce(getattr, names[:-1], module)
    setattr(parent_module, names[-1], new_module)
