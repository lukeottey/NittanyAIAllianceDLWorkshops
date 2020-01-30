from .datasets import *
from .utils import *
from .partial_datasets import *

import torchvision.transforms as transforms

num_classes_lookup_dict = {
    'mnist': 10,
    'fashion_mnist': 10,
    'kmnist': 10,
    'emnist': lambda split: {'byclass': 62, 'bymerge': 47, 'balanced': 47, 'digits': 10, 'letters': 37, 'mnist': 10}[split],
    'qmnist': 10,
    'stl10': 10,
    'cifar10': 10,
    'cifar100': 100,
    'svhn': 10,
    'svhn_extended': 10,
}

normalization_params_lookup = {
    'mnist': {'mean': [0.1307], 'std': [0.3081]},
    'stl10': {'mean': [0.4467, 0.4398, 0.4066], 'std': [0.2242, 0.2215, 0.2239]},
    'cifar10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'cifar100': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'svhn': {'mean': [0.43098, 0.430196, 0.446275], 'std': [0.196471, 0.198431, 0.199216]},
    'svhn_extended': {'mean': [0.43098, 0.430196, 0.446275], 'std': [0.196471, 0.198431, 0.199216]},
}

def build_transforms(dataset):
    if 'mnist' in dataset:
        transform = transform_test = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize(**normalization_params_lookup['mnist'])])
    elif dataset == 'stl10':
        transform = transforms.Compose([
            transforms.RandomCrop(96, padding=12),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(**normalization_params_lookup['stl10'])])
        transform_test = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize(**normalization_params_lookup['stl10'])])
    else:
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(**normalization_params_lookup[dataset])])
        transform_test = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize(**normalization_params_lookup[dataset])])
    return transform, transform_test
