import os.path as osp
import os
import yaml
import argparse
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import resize_boxes

from tracktor.datasets.mot17_tracks_wrapper import MOT17TracksWrapper, tracks_wrapper_collate

from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.motion.model_v2 import MotionModelV2
from tracktor.motion.utils import two_p_to_wh, wh_to_two_p

from tracktor.config import cfg

def get_features(obj_detect, img_list, gts):
    """
    Input:
        -img_list: list of (1, 3, w, h). Can be different sizes. 
        -gts: (batch, 4)
    Output:
        -box_features: (batch, 256, 7, 7) CUDA
        -box_head_features: (batch, 1024) CUDA
    """
    box_features_list = []
    box_head_features_list = []

    with torch.no_grad():
        gts = gts.cuda()
        for i, img in enumerate(img_list):
            obj_detect.load_image(img)

            gt = gts[i].unsqueeze(0)
            gt = resize_boxes(gt, obj_detect.original_image_sizes[0], obj_detect.preprocessed_images.image_sizes[0])
            gt = [gt]

            box_features = obj_detect.roi_heads.box_roi_pool(obj_detect.features, gt, obj_detect.preprocessed_images.image_sizes)
            box_head_features = obj_detect.roi_heads.box_head(box_features)
            box_features_list.append(box_features.squeeze(0))
            box_head_features_list.append(box_head_features.squeeze(0))

    return torch.stack(box_features_list, 0), torch.stack(box_head_features_list, 0)



def test_motion_model(dataset, tracker_config, use_ecc):
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=tracks_wrapper_collate)

    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(tracker_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))
    obj_detect.eval()
    obj_detect.cuda()

    motion_model = MotionModelV2(vis_conv_only=False, use_modulator=False, use_bn=False)
    motion_model.load_state_dict(torch.load('/home/tianjliu/MSc_project/output/tracktor/motion/motion_ecc_novisloss_nomod_nobn_jitter/motion_model_epoch_5.pth'))

    motion_model.eval()
    motion_model.cuda()

    loss_func = nn.SmoothL1Loss()

    loss_iters = []
    total_iters = len(val_loader)
    n_iters = 0

    with torch.no_grad():
        for data, label in val_loader:
            n_iters += 1
            conv_features, repr_features = get_features(obj_detect, data['curr_img'], data['curr_gt'])

            prev_loc = (data['prev_gt_warped'] if use_ecc else data['prev_gt']).cuda()
            curr_loc = (data['curr_gt_warped'] if use_ecc else data['curr_gt']).cuda()
            label_loc = label['label_gt'].cuda()
            curr_vis = data['curr_vis'].cuda()

            pred_loc_wh, vis = motion_model(conv_features, repr_features, prev_loc, curr_loc)

            pred_loc = wh_to_two_p(pred_loc_wh)
            loss = loss_func(pred_loc, label_loc)

            loss_iters.append(loss.item())

            if n_iters % 500 == 0:
                print('Iter %5d/%5d finished. Current loss %.6f.' % (n_iters, total_iters, np.mean(loss_iters)), flush=True)

    mean_loss = np.mean(loss_iters)
    print('\nAll finished! Loss %.6f' % mean_loss)

    return mean_loss

def test_tracktor_pp_motion(dataset, tracker_config, use_ecc, use_constant_v, use_bbox_regression):
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=tracks_wrapper_collate)

    if use_bbox_regression:
        obj_detect = FRCNN_FPN(num_classes=2)
        obj_detect.load_state_dict(torch.load(tracker_config['tracktor']['obj_detect_model'],
                                   map_location=lambda storage, loc: storage))
        obj_detect.eval()
        obj_detect.cuda()

    loss_func = nn.SmoothL1Loss()

    loss_iters = []
    total_iters = len(val_loader)
    n_iters = 0

    with torch.no_grad():
        for data, label in val_loader:
            n_iters += 1

            prev_loc = (data['prev_gt_warped'].cuda() if use_ecc else data['prev_gt'].cuda())
            curr_loc = (data['curr_gt_warped'].cuda() if use_ecc else data['curr_gt'].cuda())
            label_loc = label['label_gt'].cuda()

            pred_loc = curr_loc.clone()

            if use_constant_v:
                last_motion = curr_loc - prev_loc
                pred_loc += last_motion

            if use_bbox_regression:
                obj_detect.load_image(label['label_img'][0])
                pred_loc, _ = obj_detect.predict_boxes(pred_loc)

            loss = loss_func(pred_loc, label_loc)

            loss_iters.append(loss.item())

            if n_iters % 500 == 0:
                print('Iter %5d/%5d finished. Current loss %.6f.' % (n_iters, total_iters, np.mean(loss_iters)), flush=True)

    mean_loss = np.mean(loss_iters)
    print('\nAll finished! Loss %.6f' % mean_loss)

    return mean_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap', type=int, default=1)

    args = parser.parse_args()

    random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    np.random.seed(12345)
    torch.backends.cudnn.deterministic = True


    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    val_set = MOT17TracksWrapper('val', 0.8, 0.1, input_track_len=args.gap+1, 
        max_sample_frame=args.gap, get_data_mode='sample,ecc', tracker_cfg=tracker_config, val_sample=False, val_frame_gap=args.gap)

    with open(osp.join(cfg.ROOT_DIR, 'output', 'precomputed_ecc_matrices_3.pkl'), 'rb') as f:
        ecc_dict = pickle.load(f)

    val_set.load_precomputed_ecc_warp_matrices(ecc_dict)



    test_motion_model(val_set, tracker_config, use_ecc=True)

    # test_tracktor_pp_motion(val_set, tracker_config, use_ecc=True, use_constant_v=True, use_bbox_regression=True)