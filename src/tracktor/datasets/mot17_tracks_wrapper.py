import math
from random import randint
import re

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch._six import container_abcs, string_classes, int_classes

import cv2

from .mot17_tracks import MOT17Tracks

np_str_obj_array_pattern = re.compile(r'[SaUO]')

class MOT17TracksWrapper(Dataset):
    """
    Wrapper class for combining different MOT17 sequences into one dataset for the MOT17Tracks
      dataset.
    """

    def __init__(self, split, train_ratio, vis_threshold, input_track_len, max_sample_frame, 
                keep_short_track=False, train_bbox_transform='jitter', get_data_mode='raw', tracker_cfg=None):
        assert max_sample_frame > 0, 'The number of maximum previous frame to be sampled must be greater than zero!'
        assert input_track_len > max_sample_frame, 'Input track length must be greater than max_sample_frame!'

        train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                         'MOT17-11', 'MOT17-13']

        self._split = split
        self._max_sample_frame = max_sample_frame

        # if False, perform transformations according to flags (e.g., random frame sampling, ECC)
        self._get_raw_data = (get_data_mode == 'raw')
        self._random_frame_sampling = ('sample' in get_data_mode)
        self._ecc = ('ecc' in get_data_mode)

        if self._ecc:
            assert tracker_cfg is not None
            tracker_cfg = tracker_cfg['tracktor']['tracker']
            self.warp_mode = eval(tracker_cfg['warp_mode'])
            self.number_of_iterations = tracker_cfg['number_of_iterations']
            self.termination_eps = tracker_cfg['termination_eps']

            # build warp_matrix buffer to avoid repeat calculation
            self.warp_matrix_buffer = {}



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
                        sample['seq'] = seq_name
                        label['seq'] = seq_name

                        self._track_data.append(sample)
                        self._track_label.append(label)
            elif split == 'val':
                for sample in track_set._track_data:
                    if sample['start_frame'] >= val_start_frame:
                        sample, label = self.get_label_from_track(sample)
                        sample['seq'] = seq_name
                        label['seq'] = seq_name

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

    def ecc_align(self, im1, im2):
        """
        im1, im2: Tensor of shape (3, h, w)
        """
        im1 = np.transpose(im1.numpy(), (1, 2, 0))
        im2 = np.transpose(im2.numpy(), (1, 2, 0))
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                    self.number_of_iterations, self.termination_eps)
        cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
        
        return warp_matrix

    def warp_bbox(self, bbox, warp_matrix):
        """
        bbox: ndarray of shape (4, ) of two point coordinates
        warp_matrix: ndarray of shape (2, 3)
        """
        p1 = np.array([bbox[0], bbox[1], 1.0], dtype=np.float32)
        p2 = np.array([bbox[2], bbox[3], 1.0], dtype=np.float32)
        p1_n = warp_matrix @ p1
        p2_n = warp_matrix @ p2

        return np.concatenate([p1_n, p2_n], axis=0)


    def __len__(self):
        return len(self._track_data)

    def __getitem__(self, idx):
        data = self._track_data[idx]
        label = self._track_label[idx]

        if self._get_raw_data:
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
                'last_frame' : data['last_frame'],
                'seq' : data['seq']
            }

            label = {
                'gt' : torch.tensor(label['gt'], dtype=torch.float32),
                'img' : label_imgs,
                'vis' : torch.tensor(label['vis'], dtype=torch.float32),
                'start_frame' : label['start_frame'],
                'last_frame' : label['last_frame'],
                'seq' : label['seq']
            }
        else:
            # perform transformations according to flags
            if self._random_frame_sampling:
                frame_offset = randint(1, self._max_sample_frame)

                prev_gt = data['gt'][-frame_offset - 1]
                prev_img = self.image_transform(Image.open(data['im_path'][-frame_offset - 1]).convert('RGB'))
                prev_vis = data['vis'][-frame_offset - 1]
                prev_frame = data['last_frame'] - frame_offset
                prev_seq = data['seq']

                curr_gt = data['gt'][-1]
                curr_img = self.image_transform(Image.open(data['im_path'][-1]).convert('RGB'))
                curr_vis = data['vis'][-1]
                curr_frame = data['last_frame']
                curr_seq = data['seq']

                label_gt = label['gt'][frame_offset - 1]
                label_img = self.image_transform(Image.open(label['im_path'][frame_offset - 1]).convert('RGB'))
                label_vis = label['vis'][frame_offset - 1]
                label_frame = label['start_frame'] + frame_offset - 1
                label_seq = label['seq']

                if self._split == 'train' and self.train_bbox_transform is not None:
                    im_h, im_w = curr_img.size()[-2:]
                    curr_gt = self.train_bbox_transform([curr_gt], im_w, im_h)[0]

                if self._ecc:
                    prev_curr_identifier = ','.join([prev_seq, str(prev_frame), curr_seq, str(curr_frame)])
                    # if already calculated, just take out
                    if prev_curr_identifier in self.warp_matrix_buffer:
                        prev_warp_matrix = self.warp_matrix_buffer[prev_curr_identifier]
                    else:
                        prev_warp_matrix = self.ecc_align(prev_img, curr_img)
                        self.warp_matrix_buffer[prev_curr_identifier] = prev_warp_matrix
                    prev_gt_warped = self.warp_bbox(prev_gt, prev_warp_matrix)

                    curr_label_identifier = ','.join([curr_seq, str(curr_frame), label_seq, str(label_frame)])
                    if curr_label_identifier in self.warp_matrix_buffer:
                        curr_warp_matrix = self.warp_matrix_buffer[curr_label_identifier]
                    else:
                        curr_warp_matrix = self.ecc_align(curr_img, label_img)
                        self.warp_matrix_buffer[curr_label_identifier] = curr_warp_matrix
                    curr_gt_warped = self.warp_bbox(curr_gt, curr_warp_matrix)
                else:
                    prev_gt_warped, curr_gt_warped = [], []

                data = {
                    'prev_gt': prev_gt, 'prev_gt_warped': prev_gt_warped,
                    'prev_img': prev_img.unsqueeze(0), 'prev_vis': prev_vis, 'prev_frame': prev_frame, 'prev_seq': prev_seq,
                    'curr_gt': curr_gt, 'curr_gt_warped': curr_gt_warped,
                    'curr_img': curr_img.unsqueeze(0), 'curr_vis': curr_vis, 'curr_frame': curr_frame, 'curr_seq': curr_seq
                }

                label = {
                    'label_gt': label_gt, 'label_img': label_img.unsqueeze(0),
                    'label_vis': label_vis, 'label_frame': label_frame, 'label_seq': label_seq
                }
            else:
                data, label = {}, {}

        return data, label

def tracks_wrapper_collate(batch):
    """
        Based on the original pytorch default_collate function. 

        When fetched with Dataloader, the shapes of RAW data are:
        -data:
            img: list of (input_track_len, 3, h, w)
            gt:  (batch, input_track_len, 4)
            vis: (batch, input_track_len)
        -label:
            img: list of (max_sample_frame, 3, h, w)
            gt:  (batch, max_sample_frame, 4)
            vis: (batch, max_sample_frame)

        If required random frame sampling:
        -data/label:
            *_img: list of (1, 3, h, w)
            *_gt:  (batch, 4)
            *_gt_warped: (batch, 4)
            *_vis: (batch,)
    """


    default_collate_err_msg_format = "tracks_wrapper_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return tracks_wrapper_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        batch_dict = {}
        for key in elem:
            if 'img' in key:
                # do not further collate image tensors since they might be of different shapes
                batch_dict[key] = [d[key] for d in batch]
            else:
                batch_dict[key] = tracks_wrapper_collate([d[key] for d in batch])
        return batch_dict
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(tracks_wrapper_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [tracks_wrapper_collate(samples) for samples in transposed]


    raise TypeError(default_collate_err_msg_format.format(elem_type))