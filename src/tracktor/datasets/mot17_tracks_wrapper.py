import math

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .mot17_tracks import MOT17Tracks

class MOT17TracksWrapper(Dataset):
    """
    Wrapper class for combining different MOT17 sequences into one dataset for the MOT17Tracks
      dataset.
    """

    def __init__(self, split, train_ratio, vis_threshold, input_track_len, keep_short_track=False, train_bbox_transform='jitter'):
        assert input_track_len > 0, 'Input track length must be greater than zero!'

        train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                         'MOT17-11', 'MOT17-13']

        self._split = split

        self.image_transform = ToTensor()
        self.train_bbox_transform = self.bbox_jitter if train_bbox_transform == 'jitter' else None

        self._track_data = []
        self._track_label = []

        for seq_name in train_folders:
            track_set = MOT17Tracks(seq_name, vis_threshold, input_track_len + 1, keep_short_track)
            val_start_frame = int(math.floor(track_set._num_frames * train_ratio))

            if split == 'train':
                for sample in track_set._track_data:
                    if sample['start_frame'] < val_start_frame:
                        sample, label = self.get_label_from_track(sample)
                        self._track_data.append(sample)
                        self._track_label.append(label)
            elif split == 'val':
                for sample in track_set._track_data:
                    if sample['start_frame'] >= val_start_frame:
                        sample, label = self.get_label_from_track(sample)
                        self._track_data.append(sample)
                        self._track_label.append(label)

    def get_label_from_track(self, sample):
        """
        Split the last item from the track sample 
          as the corresponding label.
        """
        label = {
            'gt' : sample['gt'][-1],
            'im_path' : sample['im_path'][-1],
            'vis' : sample['vis'][-1],
            'frame' : sample['last_frame']
        }

        sample = {
            'gt' : sample['gt'][:-1],
            'im_path' : sample['im_path'][:-1],
            'vis' : sample['vis'][:-1],
            'start_frame' : sample['start_frame'],
            'last_frame' : sample['last_frame'] - 1
        }

        return sample, label

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

    def __len__(self):
        return len(self._track_data)

    def __getitem__(self, idx):
        data = self._track_data[idx]
        label = self._track_label[idx]

        data_imgs = [self.image_transform(Image.open(data['im_path'][i]).convert('RGB'))
                     for i in range(data['last_frame'] - data['start_frame'] + 1)]
        label_img = self.image_transform(Image.open(label['im_path']).convert('RGB'))

        if self._split == 'train' and self.train_bbox_transform is not None:
            im_h, im_w = label_img.size()[-2:]
            data_gts = self.train_bbox_transform(data['gt'], im_w, im_h)
        else:
            data_gts = data['gt']

        data = {
            'gt' : data_gts,
            'img' : data_imgs,
            'vis' : data['vis'],
            'start_frame' : data['start_frame'],
            'last_frame' : data['last_frame']
        }

        label = {
            'gt' : label['gt'],
            'img' : label_img,
            'vis' : label['vis'],
            'frame' : label['frame']
        }

        return data, label