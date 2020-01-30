import torch

import models

__all__ = ['count_params', 'load_model']


def load_model(model_type, checkpoint_path, **kwargs):
    model = models.__dict__[model_type](**kwargs)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def count_params(model):
    num_params = 0
    for param in list(model.parameters()):
        nn_ = 1
        for s in list(param.size()):
            nn_ = nn_ * s
        num_params += nn_
    return num_params
