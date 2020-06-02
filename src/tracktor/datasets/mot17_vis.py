import math

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
    def __init__(self, split, train_ratio, vis_threshold):
        train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                         'MOT17-11', 'MOT17-13']

        self._split = split

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

    def __len__(self):
        return len(self._vis_data)

    def __getitem__(self, idx):
        data = self._vis_data[idx]

        img = self.image_transform(Image.open(data['im_path']).convert('RGB'))

        data = {
            'img':img,
            'gt':torch.tensor(data['gt']),
            'vis':torch.tensor(data['vis'])
        }
        return data

