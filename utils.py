import os
import tqdm
import cv2
import numpy as np
import torch
from torch.utils import data

np.random.seed(1234)
torch.manual_seed(1234)

IMG_SIZE = 256


def collate_fn(batch):
    videos = [torch.stack([img for img in item[0]]) for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.as_tensor(labels)
    return [videos, labels]


class VideoDataset(data.Dataset):
    def __init__(self, root, transforms, split="train"):
        super(VideoDataset, self).__init__()
        self.root = root
        self.classes_file = os.path.join(root, 'classInd.txt')
        self.videos_dir = os.path.join(root, 'videos/')
        if split in ('train', 'val'):
            labels_file = os.path.join(root, 'trainlist.txt')
        else:
            labels_file = os.path.join(root, 'testlist.txt')

        self.video_names = []
        self.labels = []
        self.classes = {}
        self.split = split

        with open(self.classes_file, 'rt') as fp:
            for line in fp:
                ind, clas = line.split()
                self.classes[clas] = int(ind) - 1

        with open(labels_file, "rt") as fp:
            for line in tqdm.tqdm(fp, position=0, leave=True):
                if split in ('train', 'val'):
                    path, ind = line.split()
                    path = path.split('/')[1]
                    file_path = os.path.join(self.videos_dir, path)
                    if (split == 'val' and 'g24' in path or 'g25' in path) or (
                            split == 'train' and not ('g24' in path or 'g25' in path)):
                        self.video_names.append(file_path)
                        self.labels.append(int(ind) - 1)
                elif split == 'test':
                    clas, path = line.split('/')
                    file_path = os.path.join(self.videos_dir, path.strip())
                    label = self.classes[clas]
                    self.video_names.append(file_path)
                    self.labels.append(label)
        self.transforms = transforms

    def __getitem__(self, idx):
        vidcap = cv2.VideoCapture(self.video_names[idx])
        success, image = vidcap.read()
        count = 0
        images = []
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transforms(image)
            images.append(image)
            success, image = vidcap.read()
            count += 1
        label = self.labels[idx]
        return images, label

    def __len__(self):
        return len(self.video_names)
