import os.path as osp
import os
import yaml
import argparse
import random
from random import randint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tracktor.config import get_output_dir
from tracktor.datasets.mot17_tracks_wrapper import MOT17TracksWrapper, tracks_wrapper_collate

from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.motion.model import MotionModel


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


def train_main(max_previous_frame, use_ecc, vis_loss_ratio, lr, weight_decay, batch_size, output_dir, pretrain_vis_path, ex_name):
    random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    np.random.seed(12345)
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(output_dir, ex_name)
    log_file = osp.join(output_dir, 'epoch_log.txt')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    with open(log_file, 'w') as f:
        f.write('[Experiment name]%s\n\n' % ex_name)
        f.write('[Parameters]\n')
        f.write('max_previous_frame=%d\nuse_ecc=%r\nvis_loss_ratio=%f\nlr=%f\nweight_decay=%f\nbatch_size=%d\n\n' % 
            (max_previous_frame, use_ecc, vis_loss_ratio, lr, weight_decay, batch_size))
        f.write('[Loss log]\n')

    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    #################
    # Load Datasets #
    #################
    train_set = MOT17TracksWrapper('train', 0.8, 0.1, input_track_len=max_previous_frame+1, 
        max_sample_frame=max_previous_frame, get_data_mode='sample'+(',ecc' if use_ecc else ''), tracker_cfg=tracker_config)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=tracks_wrapper_collate)
    val_set = MOT17TracksWrapper('val', 0.8, 0.1, input_track_len=max_previous_frame+1, 
        max_sample_frame=max_previous_frame, get_data_mode='sample'+(',ecc' if use_ecc else ''), tracker_cfg=tracker_config)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=tracks_wrapper_collate)

    ########################
    # Initializing Modules #
    ########################
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(tracker_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))
    obj_detect.eval()
    obj_detect.cuda()

    motion_model = MotionModel(vis_conv_only=False)
    motion_model.load_vis_pretrained(pretrain_vis_path)

    motion_model.train()
    motion_model.cuda()

    optimizer = torch.optim.Adam(motion_model.parameters(), lr=lr, weight_decay=weight_decay)
    pred_loss_func = nn.SmoothL1Loss()
    vis_loss_func = nn.MSELoss()

    #######################
    # Training Parameters #
    #######################

    max_epochs = 100
    log_freq = 50

    train_pred_loss_epochs = []
    train_vis_loss_epochs = []
    val_pred_loss_epochs = []
    val_vis_loss_epochs = []
    lowest_val_loss = 9999999.9
    lowest_val_loss_epoch = -1

    ############
    # Training #
    ############

    for epoch in range(max_epochs):
        n_iter = 0
        train_pred_loss_iters = []
        train_vis_loss_iters = []
        val_pred_loss_iters = []
        val_vis_loss_iters = []

        for data, label in train_loader:
            conv_features, repr_features = get_features(obj_detect, data['curr_img'], data['curr_gt'])

            prev_loc = (data['prev_gt_warped'] if use_ecc else data['prev_gt']).cuda()
            curr_loc = data['curr_gt'].cuda()
            curr_loc_warped = (data['curr_gt_warped'].cuda() if use_ecc else None)
            label_loc = label['label_gt'].cuda()
            curr_vis = data['curr_vis'].cuda()

            n_iter += 1
            # TODO the output bbox should be (x,y,w,h)?
            optimizer.zero_grad()
            pred_loc, vis = motion_model(conv_features, repr_features, prev_loc, curr_loc, curr_loc_warped)
            pred_loss = pred_loss_func(pred_loc, label_loc)
            vis_loss = vis_loss_func(vis, curr_vis)
            loss = pred_loss + vis_loss_ratio * vis_loss

            loss.backward()
            optimizer.step()

            train_pred_loss_iters.append(pred_loss.item())
            train_vis_loss_iters.append(vis_loss.item())
            if n_iter % log_freq == 0:
                print('[Train Iter %5d] train pred loss %.6f, vis loss %.6f ...' % 
                    (n_iter, np.mean(train_pred_loss_iters[n_iter-log_freq:n_iter]), np.mean(train_vis_loss_iters[n_iter-log_freq:n_iter])))

        mean_train_pred_loss = np.mean(train_pred_loss_iters)
        mean_train_vis_loss = np.mean(train_vis_loss_iters)
        train_pred_loss_epochs.append(mean_train_pred_loss)
        train_vis_loss_epochs.append(mean_train_vis_loss)
        print('Train epoch %4d end.' % (epoch + 1))

        motion_model.eval()

        with torch.no_grad():
            for data, label in val_loader:
                conv_features, repr_features = get_features(obj_detect, data['curr_img'], data['curr_gt'])

                prev_loc = (data['prev_gt_warped'] if use_ecc else data['prev_gt']).cuda()
                curr_loc = data['curr_gt'].cuda()
                curr_loc_warped = (data['curr_gt_warped'].cuda() if use_ecc else None)
                label_loc = label['label_gt'].cuda()
                curr_vis = data['curr_vis'].cuda()

                pred_loc, vis = motion_model(conv_features, repr_features, prev_loc, curr_loc, curr_loc_warped)
                pred_loss = pred_loss_func(pred_loc, label_loc)
                vis_loss = vis_loss_func(vis, curr_vis)

                val_pred_loss_iters.append(pred_loss.item())
                val_vis_loss_iters.append(vis_loss.item())

        mean_val_pred_loss = np.mean(val_pred_loss_iters)
        mean_val_vis_loss = np.mean(val_vis_loss_iters)
        val_pred_loss_epochs.append(mean_val_pred_loss)
        val_vis_loss_epochs.append(mean_val_vis_loss)

        print('[Epoch %4d] train pred loss %.6f, vis loss %.6f; val pred loss %.6f, vis loss %.6f' % 
            (epoch+1, mean_train_pred_loss, mean_train_vis_loss, mean_val_pred_loss, mean_val_vis_loss))
        with open(log_file, 'a') as f:
            f.write('Epoch %4d: train pred loss %.6f, vis loss %.6f; val pred loss %.6f, vis loss %.6f\n' % 
                (epoch+1, mean_train_pred_loss, mean_train_vis_loss, mean_val_pred_loss, mean_val_vis_loss))

        motion_model.train()
        if mean_val_pred_loss < lowest_val_loss:
            lowest_val_loss, lowest_val_loss_epoch = mean_val_pred_loss, epoch + 1
            torch.save(motion_model.state_dict(), osp.join(output_dir, 'motion_model_epoch_%d.pth'%(epoch+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/cs/student/vbox/tianjliu/tracktor_output/motion_model')
    parser.add_argument('--pretrain_vis_path', type=str, default='/cs/student/vbox/tianjliu/tracktor_output/vis_model_epoch_94.pth')
    parser.add_argument('--ex_name', type=str, default='vis_default')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--max_previous_frame', type=int, default=2)
    parser.add_argument('--vis_loss_ratio', type=float, default=0.1)
    parser.add_argument('--use_ecc', action='store_true')

    args = parser.parse_args()
    print(args)

    train_main(args.max_previous_frame, args.use_ecc, args.vis_loss_ratio, args.lr, args.weight_decay, args.batch_size,
        args.output_dir, args.pretrain_vis_path, args.ex_name)