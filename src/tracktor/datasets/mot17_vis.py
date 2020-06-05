import math
from random import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .mot_sequence import MOT17Sequence

class MOT17Vis(Dataset):
    """
    Dataset for training visualization prediction sub-module.
    """
    def __init__(self, split, train_ratio, vis_threshold, train_bbox_jitter=True, random_image_flip=False):
        train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                         'MOT17-11', 'MOT17-13']

        self._split = split
        self._train_bbox_jitter = train_bbox_jitter
        self._random_image_flip = random_image_flip

        self.image_transform = ToTensor()

        self._vis_data = []

        for seq_name in train_folders:
            seq_set = MOT17Sequence(seq_name)
            val_start_frame = int(math.floor(len(seq_set.data) * train_ratio))

            if split == 'train':
                for sample in seq_set.data[:val_start_frame]:
                    sample_gt = []
                    sample_vis = []

                    for id in sample['gt'].keys():
                        sample_gt.append(sample['gt'][id])
                        sample_vis.append(sample['vis'][id])

                    self._vis_data.append({
                        'im_path':sample['im_path'],
                        'gt':sample_gt,
                        'vis':sample_vis
                    })
            elif split == 'val':
                for sample in seq_set.data[val_start_frame:]:
                    sample_gt = []
                    sample_vis = []

                    for id in sample['gt'].keys():
                        sample_gt.append(sample['gt'][id])
                        sample_vis.append(sample['vis'][id])

                    self._vis_data.append({
                        'im_path':sample['im_path'],
                        'gt':sample_gt,
                        'vis':sample_vis
                    })

    def bbox_jitter(self, bboxs, im_w, im_h):
        bboxs = np.array(bboxs, dtype=np.float32)
        track_len = bboxs.shape[0]

        bbox_w = bboxs[:, 2] - bboxs[:, 0]
        bbox_h = bboxs[:, 3] - bboxs[:, 1]

        bboxs[:, 0] += np.clip(np.random.normal(0.0, bbox_w * 0.05), -bbox_w * 0.1, bbox_w * 0.1)
        bboxs[:, 2] += np.clip(np.random.normal(0.0, bbox_w * 0.05), -bbox_w * 0.1, bbox_w * 0.1)
        bboxs[:, 1] += np.clip(np.random.normal(0.0, bbox_h * 0.05), -bbox_h * 0.1, bbox_h * 0.1)
        bboxs[:, 3] += np.clip(np.random.normal(0.0, bbox_h * 0.05), -bbox_h * 0.1, bbox_h * 0.1)

        bboxs[:, 0] = np.clip(bboxs[:, 0], 0, im_w - 1)
        bboxs[:, 1] = np.clip(bboxs[:, 1], 0, im_h - 1)
        bboxs[:, 2] = np.clip(bboxs[:, 2], 0, im_w - 1)
        bboxs[:, 3] = np.clip(bboxs[:, 3], 0, im_h - 1)

        return bboxs

    def bbox_flip(self, bboxs, im_w, im_h):
        bboxs = np.array(bboxs, dtype=np.float32)
        bbox_flip_right = im_w - 1 - bboxs[:, 0]
        bbox_flip_left = im_w - 1 - bboxs[:, 2]
        bboxs[:, 0] = bbox_flip_left
        bboxs[:, 2] = bbox_flip_right

        return bboxs


    def __len__(self):
        return len(self._vis_data)

    def __getitem__(self, idx):
        data = self._vis_data[idx]

        # img = self.image_transform(Image.open(data['im_path']).convert('RGB'))
        img = Image.open(data['im_path']).convert('RGB')

        if self._split == 'train':
            data_gt = data['gt']
            if self._random_image_flip:
                if random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    im_w, im_h = img.size
                    data_gt = self.bbox_flip(data_gt, im_w, im_h)

            img = self.image_transform(img)

            if self._train_bbox_jitter:
                im_h, im_w = img.size()[-2:]
                data_gt = self.bbox_jitter(data_gt, im_w, im_h)
        else:
            img = self.image_transform(img)
            data_gt = data['gt']

        data = {
            'img':img,
            'gt':torch.tensor(data_gt),
            'vis':torch.tensor(data['vis'])
        }
        return data

