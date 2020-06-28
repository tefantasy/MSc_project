import os.path as osp
import os
import yaml
import argparse
import pickle
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
from tracktor.motion.backbone_model import BackboneMotionModel
from tracktor.motion.utils import two_p_to_wh, bbox_jitter

from tracktor.config import cfg

def train_main(max_previous_frame, use_ecc, use_modulator, use_bn, vis_loss_ratio, no_vis_loss,
               lr, weight_decay, batch_size, output_dir, pretrain_vis_path, ex_name):
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
        f.write('max_previous_frame=%d\nuse_ecc=%r\nuse_modulator=%r\nvis_loss_ratio=%f\nno_vis_loss=%r\nlr=%f\nweight_decay=%f\nbatch_size=%d\n\n' % 
            (max_previous_frame, use_ecc, use_modulator, vis_loss_ratio, no_vis_loss, lr, weight_decay, batch_size))
        f.write('[Loss log]\n')

    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    #################
    # Load Datasets #
    #################
    train_set = MOT17TracksWrapper('train', 0.8, 0.0, input_track_len=max_previous_frame+1, 
        max_sample_frame=max_previous_frame, get_data_mode='sample'+(',ecc' if use_ecc else ''), tracker_cfg=tracker_config)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=tracks_wrapper_collate)
    val_set = MOT17TracksWrapper('val', 0.8, 0.1, input_track_len=max_previous_frame+1, 
        max_sample_frame=max_previous_frame, get_data_mode='sample'+(',ecc' if use_ecc else ''), tracker_cfg=tracker_config)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=tracks_wrapper_collate)

    with open(osp.join(cfg.ROOT_DIR, 'output', 'precomputed_ecc_matrices_3.pkl'), 'rb') as f:
        ecc_dict = pickle.load(f)

    train_set.load_precomputed_ecc_warp_matrices(ecc_dict)
    val_set.load_precomputed_ecc_warp_matrices(ecc_dict)

    ########################
    # Initializing Modules #
    ########################

    motion_model = BackboneMotionModel(tracker_config=tracker_config, vis_conv_only=False, use_modulator=use_modulator, use_bn=use_bn)
    # motion_model.load_vis_pretrained(pretrain_vis_path)
    motion_model.train()

    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    
    motion_model.backbone.apply(set_bn_eval)
    motion_model.cuda()

    optimizer = torch.optim.Adam(motion_model.parameters(), lr=lr, weight_decay=weight_decay)
    pred_loss_func = nn.SmoothL1Loss()
    vis_loss_func = nn.MSELoss()

    #######################
    # Training Parameters #
    #######################

    max_epochs = 100
    log_freq = 25

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
            images = data['curr_img']
            images = [img.cuda().squeeze(0) for img in images]

            # jitter target for getting roi features
            im_w = torch.tensor([img.size()[-1] for img in data['curr_img']], dtype=data['curr_gt'].dtype)
            im_h = torch.tensor([img.size()[-2] for img in data['curr_img']], dtype=data['curr_gt'].dtype)
            jittered_curr_gt = bbox_jitter(data['curr_gt'].clone(), im_w, im_h)

            target = jittered_curr_gt.cuda()
            target = [{"boxes": bbox.unsqueeze(0)} for bbox in target]

            prev_loc = (data['prev_gt_warped'] if use_ecc else data['prev_gt']).cuda()
            curr_loc = (data['curr_gt_warped'] if use_ecc else data['curr_gt']).cuda()
            label_loc = label['label_gt'].cuda()
            curr_vis = data['curr_vis'].cuda()
            label_loc_wh = two_p_to_wh(label_loc)

            n_iter += 1
            optimizer.zero_grad()
            pred_loc_wh, vis = motion_model(images, target, prev_loc, curr_loc)

            pred_loss = pred_loss_func(pred_loc_wh, label_loc_wh)
            vis_loss = vis_loss_func(vis, curr_vis)
            if no_vis_loss:
                loss = pred_loss
            else:
                loss = pred_loss + vis_loss_ratio * vis_loss

            loss.backward()
            optimizer.step()

            train_pred_loss_iters.append(pred_loss.item())
            train_vis_loss_iters.append(vis_loss.item())
            if n_iter % log_freq == 0:
                print('[Train Iter %5d] train pred loss %.6f, vis loss %.6f ...' % 
                    (n_iter, np.mean(train_pred_loss_iters[n_iter-log_freq:n_iter]), np.mean(train_vis_loss_iters[n_iter-log_freq:n_iter])),
                    flush=True)

        mean_train_pred_loss = np.mean(train_pred_loss_iters)
        mean_train_vis_loss = np.mean(train_vis_loss_iters)
        train_pred_loss_epochs.append(mean_train_pred_loss)
        train_vis_loss_epochs.append(mean_train_vis_loss)
        print('Train epoch %4d end.' % (epoch + 1))

        motion_model.eval()

        with torch.no_grad():
            for data, label in val_loader:
                images = data['curr_img']
                images = [img.cuda().squeeze(0) for img in images]

                # do not jitter for validation
                target = data['curr_gt'].cuda()
                target = [{"boxes": bbox.unsqueeze(0)} for bbox in target]

                prev_loc = (data['prev_gt_warped'] if use_ecc else data['prev_gt']).cuda()
                curr_loc = (data['curr_gt_warped'] if use_ecc else data['curr_gt']).cuda()
                label_loc = label['label_gt'].cuda()
                curr_vis = data['curr_vis'].cuda()
                label_loc_wh = two_p_to_wh(label_loc)

                pred_loc_wh, vis = motion_model(images, target, prev_loc, curr_loc)
                
                pred_loss = pred_loss_func(pred_loc_wh, label_loc_wh)
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
        motion_model.backbone.apply(set_bn_eval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/cs/student/vbox/tianjliu/tracktor_output/motion_model')
    parser.add_argument('--pretrain_vis_path', type=str, default='/cs/student/vbox/tianjliu/tracktor_output/vis_model_epoch_94.pth')
    parser.add_argument('--ex_name', type=str, default='backbone_motion_default')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--max_previous_frame', type=int, default=2)
    parser.add_argument('--vis_loss_ratio', type=float, default=1.0)
    parser.add_argument('--use_ecc', action='store_true')
    parser.add_argument('--use_modulator', action='store_true')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--no_vis_loss', action='store_true')

    args = parser.parse_args()
    print(args)

    train_main(args.max_previous_frame, 
        args.use_ecc, args.use_modulator, args.use_bn, args.vis_loss_ratio, args.no_vis_loss,
        args.lr, args.weight_decay, args.batch_size,
        args.output_dir, args.pretrain_vis_path, args.ex_name)