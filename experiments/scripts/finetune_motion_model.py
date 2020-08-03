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

from tracktor.motion.utils import two_p_to_wh

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

def train_main(vis_no_reid, no_vis_model, no_motion_repr, no_modulator, use_vis_feature_for_mod, max_sample_frame, 
               sgd, lr, weight_decay, batch_size, pretrained_path, output_dir, ex_name):
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
        f.write('[Experiment name]%s\n' % ex_name)
        f.write('[Pretrained]%s\n\n' % pretrained_path)
        f.write('[Parameters]\n')
        f.write('vis_no_reid=%r\nno_vis_model=%r\nno_motion_repr=%r\nno_modulator=%r\nuse_vis_feature_for_mod=%r\nmax_sample_frame=%d\nlr=%f\nweight_decay=%f\nbatch_size=%d\n\n' % 
            (vis_no_reid, no_vis_model, no_motion_repr, no_modulator, use_vis_feature_for_mod, max_sample_frame, lr, weight_decay, batch_size))
        f.write('[Loss log]\n')

    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    #################
    # Load Datasets #
    #################
    train_set = MOT17SimpleReIDWrapper('train', 0.8, 0.0, max_sample_frame, tracker_cfg=tracker_config)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=simple_reid_wrapper_collate)
    val_set = MOT17SimpleReIDWrapper('val', 0.8, 0.0, max_sample_frame, tracker_cfg=tracker_config)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=simple_reid_wrapper_collate)

    with open(osp.join(cfg.ROOT_DIR, 'output', 'precomputed_ecc_matrices_3.pkl'), 'rb') as f:
        ecc_dict = pickle.load(f)

    train_set.load_precomputed_ecc_warp_matrices(ecc_dict)
    val_set.load_precomputed_ecc_warp_matrices(ecc_dict)

    ########################
    # Initializing Modules #
    ########################
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

    if not no_vis_model:
        if vis_no_reid:
            vis_model = VisEst(conv_only=False)
        else:
            vis_model = VisSimpleReID()
        vis_model.load_state_dict(torch.load(pretrained_path,
                                  map_location=lambda storage, loc: storage))
    else:
        vis_model = None

    motion_model = MotionModelV3(vis_model, no_modulator=no_modulator, 
                           use_vis_model=(not no_vis_model), use_motion_repr=(not no_motion_repr), use_vis_feature_for_mod=use_vis_feature_for_mod)

    motion_model.train()
    motion_model.cuda()

    # freeze bn
    if not no_vis_model and vis_no_reid:
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        motion_model.vis_model.apply(set_bn_eval)

    if sgd:
        optimizer = torch.optim.SGD(motion_model.get_trainable_parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(motion_model.get_trainable_parameters(), lr=lr, weight_decay=weight_decay)

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
    last_save_epoch = 0
    save_epoch_freq = 5

    ############
    # Training #
    ############
    for epoch in range(max_epochs):
        n_iter = 0
        new_lowest_flag = False
        train_pred_loss_iters = []
        train_vis_loss_iters = []
        val_pred_loss_iters = []
        val_vis_loss_iters = []

        for data in train_loader:
            with torch.no_grad():
                early_reid = get_batch_mean_early_reid(reid_network, data['early_reid_patches'])
                curr_reid = reid_network(data['curr_reid_patch'].cuda())
                conv_features, repr_features = get_features(obj_detect, data['curr_img'], data['curr_gt_app'])

            prev_loc = data['prev_gt_warped'].cuda()
            curr_loc = data['curr_gt_warped'].cuda()
            label_loc = data['label_gt'].cuda()
            curr_vis = data['curr_vis'].cuda()

            n_iter += 1
            optimizer.zero_grad()

            pred_loc_wh, vis = motion_model(early_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc)
            label_loc_wh = two_p_to_wh(label_loc)

            pred_loss = pred_loss_func(pred_loc_wh, label_loc_wh)
            if not no_vis_model:
                vis_loss = vis_loss_func(vis, curr_vis)

            pred_loss.backward()
            optimizer.step()

            train_pred_loss_iters.append(pred_loss.item())
            if not no_vis_model: train_vis_loss_iters.append(vis_loss.item())
            if n_iter % log_freq == 0:
                if no_vis_model:
                    print('[Train Iter %5d] train pred loss %.6f' % (n_iter, np.mean(train_pred_loss_iters[n_iter-log_freq:n_iter])), flush=True)
                else:
                    print('[Train Iter %5d] train pred loss %.6f, vis loss %.6f ...' % 
                         (n_iter, np.mean(train_pred_loss_iters[n_iter-log_freq:n_iter]), np.mean(train_vis_loss_iters[n_iter-log_freq:n_iter])),
                         flush=True)

        mean_train_pred_loss = np.mean(train_pred_loss_iters)
        train_pred_loss_epochs.append(mean_train_pred_loss)
        if not no_vis_model:
            mean_train_vis_loss = np.mean(train_vis_loss_iters)
            train_vis_loss_epochs.append(mean_train_vis_loss)
        print('Train epoch %4d end.' % (epoch + 1))

        motion_model.eval()

        with torch.no_grad():
            for data in val_loader:
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
                val_pred_loss_iters.append(pred_loss.item())

                if not no_vis_model:
                    vis_loss = vis_loss_func(vis, curr_vis)
                    val_vis_loss_iters.append(vis_loss.item())

        mean_val_pred_loss = np.mean(val_pred_loss_iters)
        val_pred_loss_epochs.append(mean_val_pred_loss)
        if not no_vis_model:
            mean_val_vis_loss = np.mean(val_vis_loss_iters)
            val_vis_loss_epochs.append(mean_val_vis_loss)

        if no_vis_model:
            print('[Epoch %4d] train pred loss %.6f; val pred loss %.6f' % 
                  (epoch+1, mean_train_pred_loss, mean_val_pred_loss))
        else:
            print('[Epoch %4d] train pred loss %.6f, vis loss %.6f; val pred loss %.6f, vis loss %.6f' % 
                  (epoch+1, mean_train_pred_loss, mean_train_vis_loss, mean_val_pred_loss, mean_val_vis_loss))

        motion_model.train()

        if mean_val_pred_loss < lowest_val_loss:
            lowest_val_loss, lowest_val_loss_epoch = mean_val_pred_loss, epoch + 1
            last_save_epoch = lowest_val_loss_epoch
            new_lowest_flag = True
            torch.save(motion_model.state_dict(), osp.join(output_dir, 'finetune_motion_model_epoch_%d.pth'%(epoch+1)))
        elif epoch + 1 - last_save_epoch == save_epoch_freq:
            last_save_epoch = epoch + 1
            torch.save(motion_model.state_dict(), osp.join(output_dir, 'finetune_motion_model_epoch_%d.pth'%(epoch+1)))

        with open(log_file, 'a') as f:
            if no_vis_model:
                f.write('[Epoch %4d] train pred loss %.6f; val pred loss %.6f %s\n' % 
                        (epoch+1, mean_train_pred_loss, mean_val_pred_loss, '*' if new_lowest_flag else ''))
            else:
                f.write('Epoch %4d: train pred loss %.6f, vis loss %.6f; val pred loss %.6f, vis loss %.6f %s\n' % 
                    (epoch+1, mean_train_pred_loss, mean_train_vis_loss, mean_val_pred_loss, mean_val_vis_loss, '*' if new_lowest_flag else ''))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/cs/student/vbox/tianjliu/tracktor_output/motion_model')
    parser.add_argument('--pretrained_path', type=str, default='/cs/student/vbox/tianjliu/tracktor_output/vis_model_epoch_94.pth')
    parser.add_argument('--ex_name', type=str, default='finetune_default')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sgd', action='store_true')

    parser.add_argument('--max_sample_frame', type=int, default=2)
    parser.add_argument('--no_modulator', action='store_true')
    parser.add_argument('--use_vis_feature_for_mod', action='store_true')
    parser.add_argument('--vis_no_reid', action='store_true')
    parser.add_argument('--no_vis_model', action='store_true')
    parser.add_argument('--no_motion_repr', action='store_true')

    args = parser.parse_args()
    print(args)

    train_main(args.vis_no_reid, args.no_vis_model, args.no_motion_repr, 
               args.no_modulator, args.use_vis_feature_for_mod, args.max_sample_frame, 
               args.sgd, args.lr, args.weight_decay, args.batch_size, 
               args.pretrained_path, args.output_dir, args.ex_name)

