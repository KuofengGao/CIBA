import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
import os


def load_data_info(data_name, type, use="train"):
    data_path = []
    labels = []
    if data_name == "cifar10":
        data_info = open("data_prepare/cifar10/" + type + ".txt")
        for line in data_info:
            line_split = line.split(" ")
            data_path.append(line_split[0])
            labels.append(np.array(line_split[1:]).astype(float).tolist())
        np.random.seed(512)
        if type == "query":
            index = np.random.choice(len(data_path), 100, replace=False)
            data_path = np.array(data_path)[index].tolist()
            labels = np.array(labels)[index].tolist()
    elif data_name == "cifar100":
        data_info = open("data_prepare/cifar100/" + type + ".txt")
        for line in data_info:
            line_split = line.split(" ")
            data_path.append(line_split[0])
            labels.append(np.array(line_split[1:]).astype(float).tolist())
    elif data_name == "nuswide":
        data_info = open("data_prepare/nuswide/" + type + ".txt")
        for line in data_info:
            line_split = line.split(" ")
            data_path.append(line_split[0])
            labels.append(np.array(line_split[1:]).astype(float).tolist())
            if type == "query":
                data_path = data_path[:100]
                labels = labels[:100]
            if type == "database":
                data_path = data_path[:10000]
                labels = labels[:10000]
    elif data_name == "nuswide_20":
        data_info = open("data_prepare/nuswide_20/" + type + ".txt")
        for line in data_info:
            line_split = line.split(" ")
            data_path.append(line_split[0])
            labels.append(np.array(line_split[1:]).astype(float).tolist())
        np.random.seed(512)
        if type == "query":
            index = np.random.choice(len(data_path), 100, replace=False)
            data_path = np.array(data_path)[index].tolist()
            labels = np.array(labels)[index].tolist()
        if type == "database":
            index = np.random.choice(len(data_path), 10000, replace=False)
            data_path = np.array(data_path)[index].tolist()
            labels = np.array(labels)[index].tolist()
    elif data_name == "imagenet":
        data_info = open("data_prepare/imagenet/" + type + ".txt")
        for line in data_info:
            line_split = line.split(" ")
            data_path.append(line_split[0])
            labels.append(np.array(line_split[1:]).astype(float).tolist())
        np.random.seed(512)
        if type == "query" and use == "adv":
            index = np.random.choice(len(data_path), 100, replace=False)
            data_path = np.array(data_path)[index].tolist()
            labels = np.array(labels)[index].tolist()
        if type == "query" and use == "uap":
            index = np.random.choice(len(data_path), 100, replace=False)
            data_path = np.array(data_path)[index].tolist()
            labels = np.array(labels)[index].tolist()
        if type == "query" and use == "defense":
            index = np.random.choice(len(data_path), 100, replace=False)
            remain_index = [i for i in range(len(data_path)) if i not in index]
            data_path = np.array(data_path)[remain_index].tolist()
            labels = np.array(labels)[remain_index].tolist()
    elif data_name == "coco":
        data_info = open("data_prepare/coco/" + type + ".txt")
        for line in data_info:
            line_split = line.split(" ")
            data_path.append(line_split[0])
            labels.append(np.array(line_split[1:]).astype(float).tolist())
    elif data_name == "flickr":
        data_info = open("data_prepare/flickr/" + type + ".txt")
        for line in data_info:
            line_split = line.split(" ")
            data_path.append(line_split[0])
            labels.append(np.array(line_split[1:]).astype(float).tolist())
    elif data_name == "places365":
        data_info = open("data_prepare/places365/" + type + ".txt")
        for line in data_info:
            line_split = line.split(" ")
            data_path.append(line_split[0])
            labels.append(np.array(line_split[1:]).astype(float).tolist())
    return data_path, labels


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class Dataset(data.Dataset):
    def __init__(self, data_name="cifar10", type="train", use="train"):

        self.data_name = data_name
        self.type = type

        self.data_list, self.labels = load_data_info(data_name, type, use)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.data_list[index]).convert('RGB'))
        label = torch.tensor(self.labels[index])

        return img, torch.tensor(label)

    def transform(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.type == "train":
            return transforms.Compose([
                ResizeImage(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])(img)
        else:
            start_center = (256 - 224 - 1) / 2
            return transforms.Compose([
                ResizeImage(256),
                PlaceCrop(224, start_center, start_center),
                transforms.ToTensor(),
                normalize
            ])(img)


class CleanDataset(data.Dataset):
    def __init__(self, data_path, labels, random_index, data_name="imagenet", type="train"):
        self.data_name = data_name
        self.type = type

        self.data_list = np.array(data_path)[random_index].tolist()
        self.labels = np.array(labels)[random_index].tolist()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.data_list[index]).convert('RGB'))
        label = torch.tensor(self.labels[index])
        return img, label

    def transform(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.type == "train":
            return transforms.Compose([
                ResizeImage(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])(img)
        else:
            start_center = (256 - 224 - 1) / 2
            return transforms.Compose([
                ResizeImage(256),
                PlaceCrop(224, start_center, start_center),
                transforms.ToTensor(),
                normalize
            ])(img)


class PoisonDataset(data.Dataset):
    def __init__(self, args, data_path, random_index, data_name="imagenet", type="train"):
        self.data_name = data_name
        self.type = type
        self.trigger_size = args.trigger_size

        trigger = np.load(os.path.join(args.path, str(args.target_label), str(args.trigger_size), 'trigger.npy'))
        im = trigger[0]
        im = im.transpose((1, 2, 0))
        self.trigger = im
        self.data_list = np.array(data_path)[random_index].tolist()

        file = np.loadtxt(os.path.join(args.path, str(args.target_label), "target_class.txt"))
        target_label = file[:]
        self.target_label = target_label.tolist()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.data_list[index]).convert('RGB'))
        img = np.array(img)
        l = img.shape[0]
        img[l-self.trigger_size:, l-self.trigger_size:, :] = self.trigger
        return self.universal_transform(img), torch.tensor(self.target_label)

    def transform(self, img):
        if self.type == "train":
            return transforms.Compose([
                ResizeImage(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ])(img)
        else:
            start_center = (256 - 224 - 1) / 2
            return transforms.Compose([
                ResizeImage(256),
                PlaceCrop(224, start_center, start_center),
            ])(img)

    def universal_transform(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])(img)


class AdversarialDataset(data.Dataset):
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.tensor(self.data_list[index])
        label = torch.tensor(self.labels[index])
        return img, label


class TestPoisonDataset(data.Dataset):
    def __init__(self, args, data_path, labels, random_index, data_name="imagenet"):
        self.data_name = data_name
        self.trigger_size = args.trigger_size

        trigger = np.load(os.path.join(args.path, str(args.target_label), str(args.trigger_size), 'trigger.npy'))
        im = trigger[0]
        im = im.transpose((1, 2, 0))
        self.trigger = im

        self.data_list = np.array(data_path)[random_index].tolist()
        self.labels = np.array(labels)[random_index].tolist()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.data_list[index]).convert('RGB'))
        img = np.array(img)
        l = img.shape[0]
        img[l-self.trigger_size:, l-self.trigger_size:, :] = self.trigger
        label = torch.tensor(self.labels[index])
        return self.universal_transform(img), label

    def transform(self, img):
        start_center = (256 - 224 - 1) / 2
        return transforms.Compose([
            ResizeImage(256),
            PlaceCrop(224, start_center, start_center),
        ])(img)

    def universal_transform(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])(img)


class TriggerDataset(data.Dataset):
    def __init__(self, data_path, labels, random_index, data_name):
        self.data_list = np.array(data_path)[random_index].tolist()
        self.labels = np.array(labels)[random_index].tolist()
        self.data_name = data_name

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.data_list[index]).convert('RGB'))
        label = torch.tensor(self.labels[index])

        return img, torch.tensor(label)

    def transform(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        start_center = (256 - 224 - 1) / 2
        return transforms.Compose([
            ResizeImage(256),
            PlaceCrop(224, start_center, start_center),
            transforms.ToTensor(),
            normalize
        ])(img)


class HashDataset(data.Dataset):
    def __init__(self, data_name="imagenet", mode="train"):
        self.data_name = data_name

        data_path = []
        labels = []

        data_info = open(os.path.join("data_prepare", data_name, mode + ".txt"))

        for line in data_info:
            line_split = line.split(" ")
            data_path.append(line_split[0])
            labels.append(np.array(line_split[1:]).astype(float).tolist())

        self.data_list = np.array(data_path).tolist()
        self.labels = np.array(labels).tolist()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.data_list[index]).convert('RGB'))
        label = torch.tensor(self.labels[index])

        return img, label

    def transform(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        start_center = (256 - 224 - 1) / 2
        return transforms.Compose([
            ResizeImage(256),
            PlaceCrop(224, start_center, start_center),
            transforms.ToTensor(),
            normalize
        ])(img)

