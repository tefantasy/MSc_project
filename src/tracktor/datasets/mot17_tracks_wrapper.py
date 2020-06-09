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

    def __init__(self, split, train_ratio, vis_threshold, input_track_len, max_sample_frame, 
                keep_short_track=False, train_bbox_transform='jitter'):
        assert max_sample_frame > 0, 'The number of maximum previous frame to be sampled must be greater than zero!'
        assert input_track_len >= max_sample_frame, 'Input track length must be no less than max_sample_frame!'

        train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                         'MOT17-11', 'MOT17-13']

        self._split = split
        self._max_sample_frame = max_sample_frame

        self.image_transform = ToTensor()
        self.train_bbox_transform = self.bbox_jitter if train_bbox_transform == 'jitter' else None

        self._track_data = []
        self._track_label = []

        for seq_name in train_folders:
            track_set = MOT17Tracks(seq_name, vis_threshold, input_track_len + max_sample_frame, keep_short_track)
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
        Split the last max_sample_frame items from the track sample 
        as the corresponding labels.
        """
        label = {
            'gt' : sample['gt'][-self._max_sample_frame:],
            'im_path' : sample['im_path'][-self._max_sample_frame:],
            'vis' : sample['vis'][-self._max_sample_frame:],
            'start_frame' : sample['last_frame'] - self._max_sample_frame + 1,
            'last_frame' : sample['last_frame']
        }

        sample = {
            'gt' : sample['gt'][:-self._max_sample_frame],
            'im_path' : sample['im_path'][:-self._max_sample_frame],
            'vis' : sample['vis'][:-self._max_sample_frame],
            'start_frame' : sample['start_frame'],
            'last_frame' : sample['last_frame'] - self._max_sample_frame
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
        """
        When fetched with Dataloader:
        -data:
            img: (batch, input_track_len, 3, h, w)
            gt:  (batch, input_track_len, 4)
            vis: (batch, input_track_len)
        -label:
            img: (batch, max_sample_frame, 3, h, w)
            gt:  (batch, max_sample_frame, 4)
            vis: (batch, max_sample_frame)
        """
        data = self._track_data[idx]
        label = self._track_label[idx]

        data_imgs = torch.stack([self.image_transform(Image.open(data['im_path'][i]).convert('RGB'))
                     for i in range(data['last_frame'] - data['start_frame'] + 1)])
        label_imgs = torch.stack([self.image_transform(Image.open(label['im_path'][i]).convert('RGB'))
                     for i in range(label['last_frame'] - label['start_frame'] + 1)])

        if self._split == 'train' and self.train_bbox_transform is not None:
            im_h, im_w = data_imgs.size()[-2:]
            data_gts = self.train_bbox_transform(data['gt'], im_w, im_h)
        else:
            data_gts = data['gt']

        data = {
            'gt' : torch.tensor(data_gts, dtype=torch.float32),
            'img' : data_imgs,
            'vis' : torch.tensor(data['vis'], dtype=torch.float32),
            'start_frame' : data['start_frame'],
            'last_frame' : data['last_frame']
        }

        label = {
            'gt' : torch.tensor(label['gt'], dtype=torch.float32),
            'img' : label_imgs,
            'vis' : torch.tensor(label['vis'], dtype=torch.float32),
            'start_frame' : label['start_frame'],
            'last_frame' : label['last_frame']
        }

        return data, label