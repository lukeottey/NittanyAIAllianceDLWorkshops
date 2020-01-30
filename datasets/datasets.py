import os

import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

__all__ = ['cifar10', 'cifar100', 'svhn', 'stl10', 'mnist', 'fashion_mnist', 'kmnist', 'emnist', 'qmnist']

_datasets = {
    'cifar10': lambda root, train, transform, download, _: torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=download),
    'cifar100': lambda root, train, transform, download, _: torchvision.datasets.CIFAR100(root=root, train=train, transform=transform, download=download),
    'svhn': lambda root, train, transform, download, _: torchvision.datasets.SVHN(root=root, split='train' if train else 'test', transform=transform, download=download),
    'stl10': lambda root, train, transform, download, _: torchvision.datasets.STL10(root=root, split='train' if train else 'test', transform=transform, download=download),
    'mnist': lambda root, train, transform, download, _: torchvision.datasets.MNIST(root=root, train=train, transform=transform, download=download),
    'fashion_mnist': lambda root, train, transform, download, _: torchvision.datasets.FashionMNIST(root=root, train=train, transform=transform, download=download),
    'kmnist': lambda root, train, transform, download, _: torchvision.datasets.KMNIST(root=root, train=train, transform=transform, download=download),
    'emnist': lambda root, train, transform, download, split: torchvision.datasets.EMNIST(root=root, train=train, split=split, transform=transform, download=download),
    'qmnist': lambda root, train, transform, download, _: torchvision.datasets.QMNIST(root=root, train=train, transform=transform, download=download)
}

def get_dataset(dataset, root, batch_size, return_datasets, **kwargs):
    if not os.path.exists(root):
        raise FileNotFoundError('Folder {} does not exist.'.format(root))
    trainset = _datasets[dataset](root, True, \
        kwargs.get('transform', transforms.Compose([transforms.ToTensor()])), kwargs.get('download', False), kwargs.get('split', None))
    testset = _datasets[dataset](root, False, \
        kwargs.get('transform', transforms.Compose([transforms.ToTensor()])), kwargs.get('download', False), kwargs.get('split', None))
    if return_datasets:
        return trainset, testset
    trainloader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, \
        pin_memory=kwargs.get('pin_memory', torch.cuda.is_available()),
        num_workers=kwargs.get('num_workers', 0 if not torch.cuda.is_available() else 1),
        drop_last=kwargs.get('drop_last', True))
    testloader = data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, \
        pin_memory=kwargs.get('pin_memory', torch.cuda.is_available()), drop_last=kwargs.get('drop_last', True),
        num_workers=kwargs.get('num_workers', 0 if not torch.cuda.is_available() else 1))
    return trainloader, testloader

def cifar10(root, batch_size, return_datasets=False, **kwargs):
    return get_dataset('cifar10', root, batch_size, return_datasets=return_datasets, **kwargs)

def cifar100(root, batch_size, return_datasets=False, **kwargs):
    return get_dataset('cifar100', root, batch_size, return_datasets=return_datasets, **kwargs)

def svhn(root, batch_size, return_datasets=False, **kwargs):
    return get_dataset('svhn', root, batch_size, return_datasets=return_datasets, **kwargs)

def stl10(root, batch_size, return_datasets=False, **kwargs):
    return get_dataset('stl10', root, batch_size, return_datasets=return_datasets, **kwargs)

def mnist(root, batch_size, return_datasets=False, **kwargs):
    return get_dataset('mnist', root, batch_size, return_datasets=return_datasets, **kwargs)

def fashion_mnist(root, batch_size, return_datasets=False, **kwargs):
    return get_dataset('fashion_mnist', root, batch_size, return_datasets=return_datasets, **kwargs)

def kmnist(root, batch_size, return_datasets=False, **kwargs):
    return get_dataset('kmnist', root, batch_size, return_datasets=return_datasets, **kwargs)

def emnist(root, batch_size, return_datasets=False, **kwargs):
    return get_dataset('emnist', root, batch_size, return_datasets=return_datasets, **kwargs)

def qmnist(root, batch_size, return_datasets=False, **kwargs):
    return get_dataset('qmnist', root, batch_size, return_datasets=return_datasets, **kwargs)
