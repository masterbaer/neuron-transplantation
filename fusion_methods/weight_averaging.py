import copy
import torch

'''
Weight averaging of multiple models as a fusion technique. 
'''


def average_weights(models: list[torch.nn.Module]) -> torch.nn.Module:
    '''
    Simple weight averaging of multiple models.
    '''
    num_models = len(models)

    average_model = copy.deepcopy(models[0])

    # fill the parameters with 0
    average_state_dict = average_model.state_dict()
    for key in average_state_dict:
        average_state_dict[key] = torch.zeros_like(average_state_dict[key])

    # sum up all parameters
    for model in models:
        model_state_dict = model.state_dict()
        for key in average_state_dict:
            average_state_dict[key] += model_state_dict[key]

    # divide by n
    for key in average_state_dict:
        average_state_dict[key] = average_state_dict[key].float() / num_models

    average_model.load_state_dict(average_state_dict)

    return average_model
