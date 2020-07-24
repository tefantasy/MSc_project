import math
from random import randint
import re

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from torch._six import container_abcs, string_classes, int_classes

import cv2

from .mot17_tracks import MOT17Tracks

np_str_obj_array_pattern = re.compile(r'[SaUO]')

class MOT17SimpleReIDWrapper(Dataset):
    def __init__(self, split, train_ratio, vis_threshold, max_sample_frame, train_random_sample=True, val_random_sample=False, 
                 train_sample_gap=1, val_sample_gap=1, ecc=True, tracker_cfg=None):
        assert max_sample_frame > 0, 'The number of maximum previous frame to be sampled must be greater than zero!'
        assert train_sample_gap <= max_sample_frame and train_sample_gap >= 1
        assert val_sample_gap <= max_sample_frame and val_sample_gap >= 1

        train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                         'MOT17-11', 'MOT17-13']

        self.split = split
        self.is_train = (split == 'train')
        self.max_sample_frame = max_sample_frame
        self.train_random_sample = train_random_sample
        self.val_random_sample = val_random_sample
        self.train_sample_gap = train_sample_gap
        self.val_sample_gap = val_sample_gap
        self.ecc = ecc

        if self.ecc:
            assert tracker_cfg is not None
            tracker_cfg = tracker_cfg['tracktor']['tracker']
            self.warp_mode = eval(tracker_cfg['warp_mode'])
            self.number_of_iterations = tracker_cfg['number_of_iterations']
            self.termination_eps = tracker_cfg['termination_eps']

            # build warp_matrix buffer to avoid repeat calculation
            self.warp_matrix_buffer = {}

        self.image_transform = ToTensor()

        self.clip_data = []
        for seq_name in train_folders:
            track_set = MOT17Tracks(seq_name, vis_threshold, 2 * max_sample_frame + 1, simple_reid=True)
            val_start_frame = int(math.floor(track_set._num_frames * train_ratio))

            if self.is_train:
                for sample in track_set._track_data:
                    if sample['last_frame'] < val_start_frame:
                        sample['seq'] = seq_name
                        self.clip_data.append(sample)
            elif split == 'val':
                for sample in track_set._track_data:
                    if sample['start_frame'] >= val_start_frame:
                        sample['seq'] = seq_name
                        self.clip_data.append(sample)

    def load_precomputed_ecc_warp_matrices(self, ecc_dict):
        self.warp_matrix_buffer = ecc_dict


    def bbox_jitter(self, bboxs, im_w, im_h, noise_scale=0.05, clip=True):
        double_noise_scale = 2 * noise_scale

        bboxs = np.array(bboxs, dtype=np.float32)

        bbox_w = bboxs[:, 2] - bboxs[:, 0]
        bbox_h = bboxs[:, 3] - bboxs[:, 1]

        bboxs[:, 0] += np.clip(np.random.normal(0.0, bbox_w * noise_scale), -bbox_w * double_noise_scale, bbox_w * double_noise_scale)
        bboxs[:, 2] += np.clip(np.random.normal(0.0, bbox_w * noise_scale), -bbox_w * double_noise_scale, bbox_w * double_noise_scale)
        bboxs[:, 1] += np.clip(np.random.normal(0.0, bbox_h * noise_scale), -bbox_h * double_noise_scale, bbox_h * double_noise_scale)
        bboxs[:, 3] += np.clip(np.random.normal(0.0, bbox_h * noise_scale), -bbox_h * double_noise_scale, bbox_h * double_noise_scale)

        if clip:
            bboxs[:, 0] = np.clip(bboxs[:, 0], 0, im_w - 1)
            bboxs[:, 1] = np.clip(bboxs[:, 1], 0, im_h - 1)
            bboxs[:, 2] = np.clip(bboxs[:, 2], 0, im_w - 1)
            bboxs[:, 3] = np.clip(bboxs[:, 3], 0, im_h - 1)

        return bboxs

    def bbox_clip_to_image(self, bboxs, im_w, im_h):
        bboxs = np.array(bboxs, dtype=np.float32)

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

    def load_reid_patches(self, img_list, gts):
        """ Assume the length of img_list and gts are the same. """
        trans = Compose([ToPILImage(), Resize((256,128)), ToTensor()])

        img_patches = []
        for i, img in enumerate(img_list):
            pos = gts[i]

            x0 = int(pos[0])
            y0 = int(pos[1])
            x1 = int(pos[2])
            y1 = int(pos[3])
            if x0 == x1:
                if x0 != 0:
                    x0 -= 1
                else:
                    x1 += 1
            if y0 == y1:
                if y0 != 0:
                    y0 -= 1
                else:
                    y1 += 1
            img = img[:, y0:y1, x0:x1]
            img = trans(img)
            img_patches.append(img)
        img_patches = torch.stack(img_patches, 0)
        return img_patches


    def __len__(self):
        return len(self.clip_data)

    def __getitem__(self, idx):
        data = self.clip_data[idx]
        seq_name = data['seq']
        start_frame = data['start_frame']

        if self.is_train:
            if self.train_random_sample:
                frame_gap = randint(1, self.max_sample_frame)
            else:
                frame_gap = self.train_sample_gap
        else:
            if self.val_random_sample:
                frame_gap = randint(1, self.max_sample_frame)
            else:
                frame_gap = self.val_sample_gap

        prev_idx = self.max_sample_frame - frame_gap
        prev_gt = data['gt'][prev_idx]
        prev_img = self.image_transform(Image.open(data['im_path'][prev_idx]).convert('RGB'))

        curr_idx = self.max_sample_frame
        curr_gt = data['gt'][curr_idx]
        curr_img = self.image_transform(Image.open(data['im_path'][curr_idx]).convert('RGB'))
        curr_vis = data['vis'][curr_idx]

        label_idx = self.max_sample_frame + frame_gap
        label_gt = data['gt'][label_idx]
        label_img = self.image_transform(Image.open(data['im_path'][label_idx]).convert('RGB'))

        early_imgs = [self.image_transform(Image.open(im_path).convert('RGB')) for im_path in data['early_im_path']]
        early_gt = data['early_gt']

        im_h, im_w = curr_img.size()[-2:]

        # jittering
        if self.is_train:
            early_gt = self.bbox_jitter(early_gt, im_w, im_h, noise_scale=0.05)
            # curr_gt_app = self.bbox_jitter([curr_gt], im_w, im_h, noise_scale=0.05)[0]
            curr_gt = self.bbox_jitter([curr_gt], im_w, im_h, noise_scale=0.05, clip=False)[0]
            curr_gt_app = self.bbox_clip_to_image([curr_gt], im_w, im_h)[0]
        else:
            early_gt = self.bbox_clip_to_image(early_gt, im_w, im_h)
            curr_gt_app = self.bbox_clip_to_image([curr_gt], im_w, im_h)[0]

        # load reid image patches
        early_reid_patches = self.load_reid_patches(early_imgs, early_gt)
        curr_reid_patch = self.load_reid_patches([curr_img], [curr_gt_app])[0]

        if self.ecc:
            prev_curr_identifier = ','.join([seq_name, str(start_frame + prev_idx), seq_name, str(start_frame + curr_idx)])
            # if already calculated, just take it out
            if prev_curr_identifier in self.warp_matrix_buffer:
                prev_warp_matrix = self.warp_matrix_buffer[prev_curr_identifier]
            else:
                prev_warp_matrix = self.ecc_align(prev_img, curr_img)
                self.warp_matrix_buffer[prev_curr_identifier] = prev_warp_matrix
            prev_gt_warped = self.warp_bbox(prev_gt, prev_warp_matrix)

            curr_label_identifier = ','.join([seq_name, str(start_frame + curr_idx), seq_name, str(start_frame + label_idx)])
            if curr_label_identifier in self.warp_matrix_buffer:
                curr_warp_matrix = self.warp_matrix_buffer[curr_label_identifier]
            else:
                curr_warp_matrix = self.ecc_align(curr_img, label_img)
                self.warp_matrix_buffer[curr_label_identifier] = curr_warp_matrix
            prev_gt_warped = self.warp_bbox(prev_gt_warped, curr_warp_matrix)
            curr_gt_warped = self.warp_bbox(curr_gt, curr_warp_matrix)
        else:
            prev_gt_warped, curr_gt_warped = [], []

        output_data = {
            'seq': seq_name, 
            'early_reid_patches': early_reid_patches,
            'prev_gt': prev_gt, 'prev_gt_warped': prev_gt_warped, 
            'curr_gt': curr_gt, 'curr_gt_warped': curr_gt_warped, 'curr_gt_app': curr_gt_app, 
            'curr_img': curr_img.unsqueeze(0), 'curr_vis': curr_vis, 'curr_reid_patch': curr_reid_patch, 
            'label_gt': label_gt, 'label_img': label_img.unsqueeze(0)
        }
        return output_data

def simple_reid_wrapper_collate(batch):
    """
        Based on the original pytorch default_collate function. 

        When fetched with Dataloader, the shapes of data are:
            -*_img: list of (1, 3, h, w)
            -*gt*:  (batch, 4)
            -*_vis: (batch,)
            -early_reid_patches: list of (early_len, 3, 256, 128) (note that early_len of each element is not necessarily the same)
            -curr_reid_patch: (batch, 3, 256, 128)
    """


    default_collate_err_msg_format = "simple_reid_wrapper_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"

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

            return simple_reid_wrapper_collate([torch.as_tensor(b) for b in batch])
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
            if 'img' in key or 'early_reid' in key:
                # do not further collate these image tensors since they might be of different shapes
                batch_dict[key] = [d[key] for d in batch]
            else:
                batch_dict[key] = simple_reid_wrapper_collate([d[key] for d in batch])
        return batch_dict
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(simple_reid_wrapper_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [simple_reid_wrapper_collate(samples) for samples in transposed]


    raise TypeError(default_collate_err_msg_format.format(elem_type))