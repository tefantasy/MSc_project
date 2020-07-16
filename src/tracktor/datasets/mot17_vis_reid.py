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

class MOT17VisReID(Dataset):
    def __init__(self, split, train_ratio, vis_threshold):
        self.split = split
        self.is_train = (split == 'train')

        train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                         'MOT17-11', 'MOT17-13']

        self.image_transform = ToTensor()

        self.vis_data = []

        for seq_name in train_folders:
            track_set = MOT17Tracks(seq_name, vis_threshold, 1, simple_reid=True)
            val_start_frame = int(math.floor(track_set._num_frames * train_ratio))

            if self.is_train:
                for sample in track_set._track_data:
                    if sample['last_frame'] < val_start_frame:
                        sample['seq'] = seq_name
                        self.vis_data.append(sample)
            elif split == 'val':
                for sample in track_set._track_data:
                    if sample['start_frame'] >= val_start_frame:
                        sample['seq'] = seq_name
                        self.vis_data.append(sample)

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

    def load_reid_patches(self, img_list, gts):
        """ Assume the length of img_list and gts are the same. """
        trans = Compose([ToPILImage(), Resize((256,128)), ToTensor()])

        img_patches = []
        for i, im in enumerate(img_list):
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
            img = im[:, y0:y1, x0:x1]
            img = trans(img)
            img_patches.append(img)
        img_patches = torch.stack(img_patches, 0)
        return img_patches

    def __len__(self):
        return len(self.vis_data)

    def __getitem__(self, idx):
        data = self.vis_data[idx]
        seq_name = data['seq']

        img = [self.image_transform(Image.open(im_path).convert('RGB')) for im_path in data['im_path']]
        gt = data['gt']
        early_imgs = [self.image_transform(Image.open(im_path).convert('RGB')) for im_path in data['early_im_path']]
        early_gt = data['early_gt']

        im_h, im_w = img[0].size()[-2:]
        
        # jittering
        if self.is_train:
            early_gt = self.bbox_jitter(early_gt, im_w, im_h, noise_scale=0.03)
            gt = self.bbox_jitter(gt, im_w, im_h, noise_scale=0.05)
        else:
            early_gt = self.bbox_clip_to_image(early_gt, im_w, im_h)
            gt = self.bbox_clip_to_image(gt, im_w, im_h)

        # load reid image patches
        early_patches = self.load_reid_patches(early_imgs, early_gt)
        curr_patch = self.load_reid_patches(img, gt)[0]

        output_data = {
            'early_patches': early_patches,
            'patches': curr_patch,
            'img': img[0],
            'gt': gt[0],
            'vis': data['vis'][0]
        }

        return output_data

def simple_reid_wrapper_collate(batch):
    """
        Based on the original pytorch default_collate function. 

        When fetched with Dataloader, the shapes of data are:
            -img: list of (3, h, w)
            -gt:  (batch, 4)
            -vis: (batch,)
            -early_patches: list of (early_len, 3, 256, 128) (note that early_len of each element is not necessarily the same)
            -patches: (batch, 3, 256, 128)
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
            if 'img' in key or 'early' in key:
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
