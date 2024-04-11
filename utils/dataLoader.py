import os
from glob import glob

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, ImageNet, ImageFolder
from utils.tools import check_folder, data_transform


class ImageDataset(Dataset):
    def __init__(self, img_size, dataset_path):
        self.train_images = self.listdir(dataset_path)
        self.train_labels = list(pd.read_csv(dataset_path + '/annotations.csv', header=None).iloc[:, 1])

        # interpolation=transforms.InterpolationMode.BICUBIC, antialias=True

        self.transform = data_transform(img_size)

    def listdir(self, dir_path):
        extensions = ['png', 'jpg']
        file_path = []
        for ext in extensions:
            file_path += glob(os.path.join(dir_path, '*.' + ext))
        file_path.sort()
        return file_path

    def __getitem__(self, index):
        sample_path = self.train_images[index]
        img = Image.open(sample_path).convert('RGB')
        img = self.transform(img)

        # 여기서 문제
        label = self.train_labels[index]

        return img, label

    def __len__(self):
        return len(self.train_images)


def download_mnist_data(train=True):
    download_path = os.path.join(os.getcwd(), 'dataset', 'mnist')
    check_folder(download_path)

    if train:
        data = MNIST(
            root=download_path,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
    else:
        data = MNIST(
            root=download_path,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

    return data


def load_imagenet(root=None, train=True):
    if root is None:
        return root

    check_folder(root)
    transfrom = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if train:
        load_root = os.path.join(root, 'train')
        data = ImageFolder(load_root, transform=transfrom)
    else:
        load_root = os.path.join(root, 'test')
        data = ImageFolder(load_root, transform=transfrom)
    return data

