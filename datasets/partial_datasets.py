import os
from glob import glob

from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PartialDataset(Dataset):
    default_transform = transforms.Compose([transforms.ToTensor()])
    def __init__(self, root, dataset, split, classes, transform=None):
        assert len(classes) <= 3 and max(classes) <= 2
        assert dataset in ('mnist_partial', 'svhn_partial')
        root = '{root}/{dataset}/{split}'.format(root=root, \
            dataset=dataset, split=split)
        if not os.path.exists(root):
            raise FileNotFoundError('{} does not exist.'.format(root))
        self.data, self.labels = [], []
        for c in classes:
            img_paths = glob('{path}/{c}/*.png'.format(path=root, c=c))
            for path in img_paths:
                self.labels.append(c)
                self.data.append(Image.open(path))
        if transform is None:
            self.transform = self.default_transform
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = self.transform(img)
        return img, label
