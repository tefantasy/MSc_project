from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import cv2

from .mot17_clips import MOT17Clips

class MOT17ClipsWrapper(Dataset):
    def __init__(self, split, train_ratio, vis_threshold, clip_len=10, min_track_len=2, train_jitter=True, ecc=True, tracker_cfg=None):
        assert min_track_len >= 2
        assert clip_len > min_track_len

        train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                         'MOT17-11', 'MOT17-13']

        self.split = split
        self.is_train = (split == 'train')
        self.train_jitter = train_jitter
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
            seq_clip_set = MOT17Clips(seq_name, split, train_ratio, vis_threshold, clip_len, min_track_len)
            self.clip_data.extend(seq_clip_set.seq_clip_data)

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

    def __len__(self):
        return len(self.clip_data)


    def __getitem__(self, idx):
        data = self.clip_data[idx]
        tracks = data['tracks']
        seq_name, start_frame = data['seq'], data['start_frame']

        img_list = [self.image_transform(Image.open(im_path).convert('RGB')) for im_path in data['im_paths']]
        im_h, im_w = img_list[0].size()[-2:]

        tracks_data = []
        for track in tracks:
            # historical data
            historical_gt = track['gt'][:-2]
            if self.is_train and self.train_jitter:
                historical_gt = self.bbox_jitter(historical_gt, im_w, im_h, noise_scale=0.05)
            else:
                historical_gt = self.bbox_clip_to_image(historical_gt, im_w, im_h)
            historical = {
                'gt': torch.tensor(historical_gt, dtype=torch.float32),
                'frame_offset': torch.tensor(track['frame_offset'][:-2], dtype=torch.long),
                'vis': torch.tensor(track['vis'][:-2], dtype=torch.float32)
            }

            # prev, curr, label
            prev_gt = track['gt'][-3]
            prev_frame_offset = track['frame_offset'][-3]

            curr_gt = track['gt'][-2]
            curr_frame_offset = track['frame_offset'][-2]
            curr_vis = track['vis'][-2]

            label_gt = track['gt'][-1]
            label_frame_offset = track['frame_offset'][-1]
            label_vis = track['vis'][-1]

            if self.is_train and self.train_jitter:
                curr_gt_app = self.bbox_jitter([curr_gt], im_w, im_h, noise_scale=0.05)[0]
                curr_gt = self.bbox_jitter([curr_gt], im_w, im_h, noise_scale=0.03, clip=False)[0]
            else:
                curr_gt_app = self.bbox_clip_to_image([curr_gt], im_w, im_h)[0]

            if self.ecc:
                prev_curr_identifier = ','.join([seq_name, str(start_frame + prev_frame_offset), seq_name, str(start_frame + curr_frame_offset)])
                # if already calculated, just take it out
                if prev_curr_identifier in self.warp_matrix_buffer:
                    prev_warp_matrix = self.warp_matrix_buffer[prev_curr_identifier]
                else:
                    prev_warp_matrix = self.ecc_align(img_list[-3], img_list[-2])
                    self.warp_matrix_buffer[prev_curr_identifier] = prev_warp_matrix
                prev_gt_warped = self.warp_bbox(prev_gt, prev_warp_matrix)

                curr_label_identifier = ','.join([seq_name, str(start_frame + curr_frame_offset), seq_name, str(start_frame + label_frame_offset)])
                if curr_label_identifier in self.warp_matrix_buffer:
                    curr_warp_matrix = self.warp_matrix_buffer[curr_label_identifier]
                else:
                    curr_warp_matrix = self.ecc_align(img_list[-2], img_list[-1])
                    self.warp_matrix_buffer[curr_label_identifier] = curr_warp_matrix
                prev_gt_warped = self.warp_bbox(prev_gt_warped, curr_warp_matrix)
                curr_gt_warped = self.warp_bbox(curr_gt, curr_warp_matrix)
            else:
                prev_gt_warped, curr_gt_warped = [], []

            track_data = {
                'historical': historical,
                'prev_gt': torch.tensor(prev_gt, dtype=torch.float32), 'prev_gt_warped': torch.tensor(prev_gt_warped, dtype=torch.float32),
                'curr_gt': torch.tensor(prev_gt, dtype=torch.float32), 'curr_gt_warped': torch.tensor(curr_gt_warped, dtype=torch.float32),
                'curr_gt_app': torch.tensor(curr_gt_app, dtype=torch.float32), 'curr_frame_offset': curr_frame_offset,
                'curr_vis': curr_vis, 
                'label_gt': torch.tensor(label_gt, dtype=torch.float32), 'label_vis': label_vis
            }
            tracks_data.append(track_data)

        # stack data of each track together
        historical = [track_data['historical'] for track_data in tracks_data]
        prev_gt = torch.stack([track_data['prev_gt'] for track_data in tracks_data])
        prev_gt_warped = torch.stack([track_data['prev_gt_warped'] for track_data in tracks_data])
        curr_gt = torch.stack([track_data['curr_gt'] for track_data in tracks_data])
        curr_gt_warped = torch.stack([track_data['curr_gt_warped'] for track_data in tracks_data])
        curr_gt_app = torch.stack([track_data['curr_gt_app'] for track_data in tracks_data])
        curr_frame_offset = torch.tensor([track_data['curr_frame_offset'] for track_data in tracks_data], dtype=torch.long)
        curr_vis = torch.tensor([track_data['curr_vis'] for track_data in tracks_data], dtype=torch.float32)
        label_gt = torch.stack([track_data['label_gt'] for track_data in tracks_data])
        label_vis = torch.tensor([track_data['label_vis'] for track_data in tracks_data], dtype=torch.float32)

        output_data = {
            'seq': seq_name,
            'start_frame': start_frame,
            'imgs': img_list,
            'historical': historical, 'prev_gt': prev_gt, 'prev_gt_warped': prev_gt_warped,
            'curr_gt': curr_gt, 'curr_gt_warped': curr_gt_warped, 'curr_gt_app': curr_gt_app, 
            'curr_frame_offset': curr_frame_offset, 'curr_vis': curr_vis,
            'label_gt': label_gt, 'label_vis': label_vis
        }
        return output_data

def clips_wrapper_collate(batch):
    """
    NOTE: only support batch_size = 1

    Output: a dict contains keys of
        -seq: str
        -start_frame: int
        -imgs: list of tensor (3, h, w), len=clip_len
        -historical: list (len=n_tracks) of dict that contains keys of
            -gt: (n_historical, 4)
            -frame_offset: (n_historical,)
            -vis: (n_historical,)
        -*gt*: (n_tracks, 4)
        -*frame_offset: (n_tracks,)
        -*vis: (n_tracks,)

    The shapes refer to Tensor data.
    """
    return batch[0]