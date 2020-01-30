import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

def find_mean_n_std(dataset, batch_size):
    """
    Finds normalization parameters (mean and std) averages across each color channel
    for the input dataset.
    Arguments:
        dataset {[torch.utils.data.Dataset]} -- [Classification Dataset]
    Returns:
        mean, std
    """
    mean = 0.
    std = 0.
    total = 0
    loader = DataLoader(dataset, batch_size=batch_size)
    for imgs, _ in tqdm(loader):
        num_samples, num_channels, h, w = imgs.size()
        imgs = imgs.view(num_samples, num_channels, h * w)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        total += num_samples
    return mean / total, std / total
