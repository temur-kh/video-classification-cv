import os
import random
import tqdm
import cv2
import numpy as np
import torch
from torch.utils import data

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)

IMG_SIZE = 256
FRAMES_CNT = 16


def set_frames_cnt(frames_cnt):
    global FRAMES_CNT
    FRAMES_CNT = frames_cnt


def collate_fn(batch):
    videos = torch.stack([img for item in batch for img in random.sample(item[0], k=FRAMES_CNT)])
    labels = [item[1] for item in batch]
    labels = torch.as_tensor(labels)
    return [videos, labels]


def collate_fn_memory(batch):
    videos = torch.stack([img for item in batch for img in item[0]])
    labels = [item[1] for item in batch]
    labels = torch.as_tensor(labels)
    return [videos, labels]


def get_collate_fn(in_memory):
    if in_memory:
        return collate_fn_memory
    else:
        return collate_fn


class VideoDataset(data.Dataset):
    def __init__(self, root, transforms, split="train", in_memory=False, stride=1, frame_cnt=16):
        super(VideoDataset, self).__init__()
        self.root = root
        self.stride = stride
        self.in_memory = in_memory
        self.frame_cnt = frame_cnt
        self.transforms = transforms

        self.classes_file = os.path.join(root, 'classInd.txt')
        self.videos_dir = os.path.join(root, 'videos/')
        if split in ('train', 'val'):
            labels_file = os.path.join(root, 'trainlist.txt')
        else:
            labels_file = os.path.join(root, 'testlist.txt')

        self.videos = []
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
                file_path = None
                if split in ('train', 'val'):
                    path, ind = line.split()
                    path = path.split('/')[1]
                    file_path = os.path.join(self.videos_dir, path)
                    if (split == 'val' and 'g24' in path or 'g25' in path) or (
                            split == 'train' and not ('g24' in path or 'g25' in path)):
                        self.labels.append(int(ind) - 1)
                    else:
                        continue
                elif split == 'test':
                    clas, path = line.split('/')
                    file_path = os.path.join(self.videos_dir, path.strip())
                    label = self.classes[clas]
                    self.labels.append(label)

                file_path = file_path.strip()
                self.video_names.append(file_path)
                if self.in_memory:
                    self.videos.append(read_video(
                        file_path, transforms=self.transforms, in_memory=in_memory, stride=stride, frame_cnt=frame_cnt))

    def __getitem__(self, idx):
        if self.in_memory:
            images = [self.transforms(item) for item in self.videos[idx]]
        else:
            images = read_video(self.video_names[idx], self.transforms)
        label = self.labels[idx]
        return images, label

    def __len__(self):
        return len(self.video_names)


def read_video(path, transforms=None, in_memory=False, stride=1, frame_cnt=16):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    images = []
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if transforms and not in_memory:
            image = transforms(image)
        images.append(image)
        success, image = vidcap.read()
        count += 1
    if in_memory:
        stride = min(int(len(images) / frame_cnt), stride)
        starting_point = random.randint(0, len(images) - stride * frame_cnt)
        images = images[starting_point:starting_point + stride * frame_cnt:stride]
    return images