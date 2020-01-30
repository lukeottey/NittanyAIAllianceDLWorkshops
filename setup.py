import os
import torch
import numpy as np
from imageio import imwrite

from datasets import mnist, svhn

root = './data'

def save_imgs(data, labels, dataset, split):
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy()
    if not isinstance(data, np.ndarray):
        data = data.numpy()
    directory = os.path.join(root, dataset.upper())
    if not os.path.exists(directory):
        os.mkdir(directory)
    subdir = os.path.join(directory, split)
    if not os.path.exists(subdir):
        os.mkdir(subdir)
        class_dirs = np.unique(labels)
        for c in class_dirs:
            os.mkdir(os.path.join(subdir, str(int(c))))
    for i, (img, label) in enumerate(zip(data, labels)):
        if len(img.shape) == 3:
            img = img.transpose((1, 2, 0))
        uri = '{subdir}/{label}/{i}.png'.format(subdir=subdir, label=label, i=i)
        imwrite(uri=uri, im=img)    

def set_mnist(directory):
    trainset, testset = mnist(root='./data', batch_size=-1, return_datasets=True, download=True, \
        transform=None, transform_test=None)

    train_idxs_1h = np.array(
        (trainset.targets == 0).type(torch.int) +
        (trainset.targets == 1).type(torch.int) +
        (trainset.targets == 2).type(torch.int))
    train_idxs = []
    for i, idx in enumerate(train_idxs_1h):
        if idx == 1:
            train_idxs.append(i)

    test_idxs_1h = np.array(
        (testset.targets == 0).type(torch.int) +
        (testset.targets == 1).type(torch.int) +
        (testset.targets == 2).type(torch.int))
    test_idxs = []
    for i, idx in enumerate(test_idxs_1h):
        if idx == 1:
            test_idxs.append(i)

    train_data = trainset.data[train_idxs]
    test_data = testset.data[test_idxs]
    train_labels = trainset.targets[train_idxs]
    test_labels = testset.targets[test_idxs]
    save_imgs(train_data, train_labels, dataset='mnist_partial', split='train')
    save_imgs(test_data, test_labels, dataset='mnist_partial', split='test')

def set_svhn(directory):
    trainset, testset = svhn(root='./data', batch_size=-1, return_datasets=True, download=True, \
        transform=None, transform_test=None)
    train_idxs_1h = np.array(
        (trainset.labels == 0).astype(int) +
        (trainset.labels == 1).astype(int) +
        (trainset.labels == 2).astype(int))
    train_idxs = []
    for i, idx in enumerate(train_idxs_1h):
        if idx == 1:
            train_idxs.append(i)

    test_idxs_1h = np.array(
        (testset.labels == 0).astype(int) +
        (testset.labels == 1).astype(int) +
        (testset.labels == 2).astype(int))
    test_idxs = []
    for i, idx in enumerate(test_idxs_1h):
        if idx == 1:
            test_idxs.append(i)
    
    train_data = trainset.data[train_idxs]
    test_data = testset.data[test_idxs]
    train_labels = trainset.labels[train_idxs]
    test_labels = testset.labels[test_idxs]
    save_imgs(train_data, train_labels, dataset='svhn_partial', split='train')
    save_imgs(test_data, test_labels, dataset='svhn_partial', split='test')

def main():
    if not os.path.exists(root):
        os.mkdir(root)
    print('----- BUILDING MNIST -----')
    mnist_dir = os.path.join(root, 'mnist_partial')
    if not os.path.exists(mnist_dir):
        os.mkdir(mnist_dir)
        set_mnist(mnist_dir)
    print('----- BUILDING SVHN -----')
    svhn_dir = os.path.join(root, 'svhn_partial')
    if not os.path.exists(svhn_dir):
        os.mkdir(svhn_dir)
        set_svhn(svhn_dir)

if __name__ == '__main__':
    main()
