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
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from torchvision.ops.boxes import clip_boxes_to_image

from tracktor.config import cfg, get_output_dir
from tracktor.datasets.mot17_simple_reid_wrapper import MOT17SimpleReIDWrapper, simple_reid_wrapper_collate

from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.motion.vis_simple_reid import VisSimpleReID
from tracktor.motion.visibility import VisEst
from tracktor.motion.model_v3 import MotionModelV3
from tracktor.reid.resnet import resnet50

from tracktor.motion.utils import two_p_to_wh, wh_to_two_p

def get_features(obj_detect, img_list, gts):
    """
    Input:
        -img_list: list of (1, 3, h, w). Can be different sizes. 
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
            gt = clip_boxes_to_image(gt, img.shape[-2:])
            gt = resize_boxes(gt, obj_detect.original_image_sizes[0], obj_detect.preprocessed_images.image_sizes[0])
            gt = [gt]

            box_features = obj_detect.roi_heads.box_roi_pool(obj_detect.features, gt, obj_detect.preprocessed_images.image_sizes)
            box_head_features = obj_detect.roi_heads.box_head(box_features)
            box_features_list.append(box_features.squeeze(0))
            box_head_features_list.append(box_head_features.squeeze(0))

    return torch.stack(box_features_list, 0), torch.stack(box_head_features_list, 0)

def get_batch_mean_early_reid(reid_model, early_reid_patches):
    with torch.no_grad():
        batch_reid_features = []
        for reid_patch in early_reid_patches:
            reid_features = reid_model(reid_patch.cuda())
            reid_features = torch.mean(reid_features, 0)
            batch_reid_features.append(reid_features)
        batch_reid_features = torch.stack(batch_reid_features, 0)
        return batch_reid_features


def test_motion_model(val_loader, tracker_config, motion_model):
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(tracker_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))
    obj_detect.eval()
    obj_detect.cuda()

    reid_network = resnet50(pretrained=False, output_dim=128)
    reid_network.load_state_dict(torch.load(tracker_config['tracktor']['reid_weights'],
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()


    pred_loss_func = nn.SmoothL1Loss()

    loss_iters = []
    low_vis_loss_sum = 0.0
    low_vis_num = 0
    high_vis_loss_sum = 0.0
    high_vis_num = 0
    total_iters = len(val_loader)
    n_iters = 0

    with torch.no_grad():
        for data in val_loader:
            n_iters += 1

            early_reid = get_batch_mean_early_reid(reid_network, data['early_reid_patches'])
            curr_reid = reid_network(data['curr_reid_patch'].cuda())
            conv_features, repr_features = get_features(obj_detect, data['curr_img'], data['curr_gt_app'])

            prev_loc = data['prev_gt_warped'].cuda()
            curr_loc = data['curr_gt_warped'].cuda()
            label_loc = data['label_gt'].cuda()
            curr_vis = data['curr_vis'].cuda()

            pred_loc_wh, vis = motion_model(early_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc)
            label_loc_wh = two_p_to_wh(label_loc)

            pred_loss = pred_loss_func(pred_loc_wh, label_loc_wh)
            loss_iters.append(pred_loss.item())

            low_vis_ind = curr_vis < 0.3
            if low_vis_ind.any():
                low_vis_pred_loss = pred_loss_func(pred_loc_wh[low_vis_ind], label_loc_wh[low_vis_ind])
                low_vis_loss_sum += (low_vis_pred_loss * torch.sum(low_vis_ind)).item()
                low_vis_num += torch.sum(low_vis_ind).item()

            high_vis_ind = curr_vis > 0.7
            if high_vis_ind.any():
                high_vis_pred_loss = pred_loss_func(pred_loc_wh[high_vis_ind], label_loc_wh[high_vis_ind])
                high_vis_loss_sum += (high_vis_pred_loss * torch.sum(high_vis_ind)).item()
                high_vis_num += torch.sum(high_vis_ind).item()

            if n_iters % 50 == 0:
                print('Iter %5d/%5d finished.' % (n_iters, total_iters), flush=True)

    mean_loss = np.mean(loss_iters)
    mean_low_vis_loss = low_vis_loss_sum / low_vis_num
    mean_high_vis_loss = high_vis_loss_sum / high_vis_num

    print('All finished! Loss %.6f, low vis loss %.6f, high vis loss %.6f.' % (mean_loss, mean_low_vis_loss, mean_high_vis_loss))

def test_tracktor_motion(val_loader, tracker_config, bbox_regression=True):
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(tracker_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))
    obj_detect.eval()
    obj_detect.cuda()

    pred_loss_func = nn.SmoothL1Loss()

    loss_iters = []
    low_vis_loss_sum = 0.0
    low_vis_num = 0
    high_vis_loss_sum = 0.0
    high_vis_num = 0
    total_iters = len(val_loader)
    n_iters = 0

    print(total_iters)

    with torch.no_grad():
        for data in val_loader:
            n_iters += 1

            prev_loc = data['prev_gt_warped'].cuda()
            curr_loc = data['curr_gt_warped'].cuda()
            label_loc = data['label_gt'].cuda()
            curr_vis = data['curr_vis'].cuda()

            pred_loc = curr_loc.clone()

            last_motion = curr_loc - prev_loc
            pred_loc += last_motion

            if bbox_regression:
                obj_detect.load_image(data['label_img'][0])
                pred_loc, _ = obj_detect.predict_boxes(pred_loc)

            label_loc_wh = two_p_to_wh(label_loc)
            pred_loc_wh = two_p_to_wh(pred_loc)

            pred_loss = pred_loss_func(pred_loc_wh, label_loc_wh)
            loss_iters.append(pred_loss.item())

            low_vis_ind = curr_vis < 0.3
            if low_vis_ind.any():
                low_vis_pred_loss = pred_loss_func(pred_loc_wh[low_vis_ind], label_loc_wh[low_vis_ind])
                low_vis_loss_sum += (low_vis_pred_loss * torch.sum(low_vis_ind)).item()
                low_vis_num += torch.sum(low_vis_ind).item()

            high_vis_ind = curr_vis > 0.7
            if high_vis_ind.any():
                high_vis_pred_loss = pred_loss_func(pred_loc_wh[high_vis_ind], label_loc_wh[high_vis_ind])
                high_vis_loss_sum += (high_vis_pred_loss * torch.sum(high_vis_ind)).item()
                high_vis_num += torch.sum(high_vis_ind).item()

            if n_iters % 500 == 0:
                print('Iter %5d/%5d finished.' % (n_iters, total_iters), flush=True)

    mean_loss = np.mean(loss_iters)
    mean_low_vis_loss = low_vis_loss_sum / low_vis_num
    mean_high_vis_loss = high_vis_loss_sum / high_vis_num

    print('All finished! Loss %.6f, low vis loss %.6f, high vis loss %.6f.' % (mean_loss, mean_low_vis_loss, mean_high_vis_loss))


if __name__ == '__main__':
    random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    np.random.seed(12345)
    torch.backends.cudnn.deterministic = True

    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    val_set = MOT17SimpleReIDWrapper('val', 0.8, 0.0, 2, tracker_cfg=tracker_config, val_sample_gap=2)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=simple_reid_wrapper_collate)

    with open(osp.join(cfg.ROOT_DIR, 'output', 'precomputed_ecc_matrices_3.pkl'), 'rb') as f:
        ecc_dict = pickle.load(f)

    val_set.load_precomputed_ecc_warp_matrices(ecc_dict)

    vis_model = VisSimpleReID()
    # vis_model = VisEst(conv_only=False)

    motion_model = MotionModelV3(vis_model, no_modulator=False, use_vis_model=True, use_motion_repr=True)
    motion_model.load_state_dict(torch.load('/cs/student/vbox/tianjliu/tracktor_output/motion_model/finetune_l21e-4_chjitter/finetune_motion_model_epoch_5.pth'))

    motion_model.eval()
    motion_model.cuda()

    # test_motion_model(val_loader, tracker_config, motion_model)

    test_tracktor_motion(val_loader, tracker_config, bbox_regression=True)