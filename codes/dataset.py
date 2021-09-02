import os
import torch
import torchvision
from pathlib import Path
from PIL import Image
from torchvision import transforms


def transform():
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
        torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    return train_transform


class DealDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels, fmt='jpg'):
        self.image_dir = Path(image_dir)
        self.trans = transform()
        self.labels = labels
        self.images = []
        for image_dir in list(self.image_dir.iterdir()):
            self.images.extend(list(image_dir.glob(f'*.{fmt}')))
        self.image_num = len(self.images)

    def __len__(self):
        return self.image_num

    def __getitem__(self, item):
        image_path = self.images[item]
        img = self.process_image(image_path)
        label = self.process_label(image_path)

        return img, label

    def process_image(self, path):
        img = Image.open(path)
        return self.trans(img)

    def process_label(self, path):
        class_name = path.parent.name
        if class_name not in self.labels:
            raise ValueError(f'图片：{path} 类别：{class_name} 不在分类标签中')
        label = self.labels.index(class_name)
        return label
