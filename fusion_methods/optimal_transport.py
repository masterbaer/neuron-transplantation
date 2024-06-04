import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fusion_methods.optimal_transport_dir.wasserstein_ensemble import geometric_ensembling_modularized_multiple_models, \
    geometric_ensembling_modularized
from fusion_methods.optimal_transport_dir.utils import get_model_activations, _get_config

import torch
from types import SimpleNamespace

'''
Optimal-Transport fusion as a fusion method. Code is taken from https://github.com/sidak/otfusion and trimmed down to
contain only the essentials to conduct the fusion. It is adapted to allow fusing multiple models, which is not
implemented in the public repository despite the usage in the paper. 
The folder optimal_transport_dir contains the code while this file sets the necessary args to run the method. 
'''


def fuse_optimal_transport(models, train_loader=None, test_loader=None, device=None, handle_skips=False):
    '''
    Optimal Transport Fusion from https://github.com/sidak/otfusion . This method simply sets the args for the adapted
    code (to support fusing multiple models). Only the weight mode is supported here (geom_ensemble_type="wts").
    Unnecessary args related to the activation mode and other experiments in the original repository are kept
    but not used.
    '''
    args = SimpleNamespace(gpu_id=-1, sweep_name="exp_sample", exact=True,
                           correction="True", ground_metric="euclidean",
                           activation_mode="raw", geom_ensemble_type="wts",
                           sweep_id="21", act_num_samples=200, ground_metric_normalize="none",
                           activation_seed=21, prelu_acts=True, past_correction=True,
                           not_squared=True, dist_normalize=True, width_ratio=1,
                           activation_histograms=False, dataset="none", handle_skips=handle_skips,
                           reg=1e-2, soft_temperature=1, ground_metric_eff=False,
                           update_acts=False, eval_aligned=False, normalize_wts=False,
                           autoencoder=False, debug=False, clip_gm=False, importance=None,
                           unbalanced=False, proper_marginals=False, ensemble_step=0.5,
                           skip_last_layer=False)

    # setattr(args, "geom_ensemble_type", "acts")
    # activations = get_model_activations(args, models)

    activations = None
    # activations not needed in weight mode
    # geometric_model = geometric_ensembling_modularized(args, models, train_loader, test_loader, activations)

    geometric_model = geometric_ensembling_modularized_multiple_models(args, models)
    #

    return geometric_model
