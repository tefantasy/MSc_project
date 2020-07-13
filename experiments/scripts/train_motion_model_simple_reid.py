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
from tracktor.motion.model_simple_reid import MotionModelSimpleReID
from tracktor.motion.model_simple_reid_v2 import MotionModelSimpleReIDV2
from tracktor.motion.refine_model import RefineModel
from tracktor.reid.resnet import resnet50

from tracktor.motion.utils import two_p_to_wh


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
            gt = clip_boxes_to_image(gt, img.shape[-2:])
            gt = resize_boxes(gt, obj_detect.original_image_sizes[0], obj_detect.preprocessed_images.image_sizes[0])
            gt = [gt]

            box_features = obj_detect.roi_heads.box_roi_pool(obj_detect.features, gt, obj_detect.preprocessed_images.image_sizes)
            box_head_features = obj_detect.roi_heads.box_head(box_features)
            box_features_list.append(box_features.squeeze(0))
            box_head_features_list.append(box_head_features.squeeze(0))

    return torch.stack(box_features_list, 0), torch.stack(box_head_features_list, 0)

def get_batch_mean_early_reid(reid_model, early_reid_patches):
    batch_reid_features = []
    for reid_patch in early_reid_patches:
        reid_features = reid_model(reid_patch.cuda())
        reid_features = torch.mean(reid_features, 0)
        batch_reid_features.append(reid_features)
    batch_reid_features = torch.stack(batch_reid_features, 0)
    return batch_reid_features



def weighted_smooth_l1_loss(pred, target, vis):
    """
    pred: (batch, 4)
    target: (batch, 4)
    vis: (batch, ) used to calculate weights
    """
    gamma = 4
    batch_size = pred.size()[0]

    loss_abs = torch.abs(pred - target)
    loss_quadratic = 0.5 * (loss_abs ** 2)

    loss = torch.where(loss_abs >= 1.0, loss_abs - 0.5, loss_quadratic)

    weights = torch.pow(gamma, 1.0 - vis).unsqueeze(-1).expand(-1, 4)
    loss = weights * loss

    # 2 for compensating the increase because of weights
    loss = torch.sum(loss) / (batch_size * 4 * 2)

    return loss



def train_main(v2, use_refine_model, use_ecc, use_modulator, use_bn, use_residual, vis_roi_features, no_visrepr, vis_loss_ratio, no_vis_loss,
               max_sample_frame, lr, weight_decay, batch_size, output_dir, ex_name):
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
        f.write('use_ecc=%r\nuse_modulator=%r\nuse_bn=%r\nuse_residual=%r\nvis_roi_features=%r\nno_visrepr=%r\nvis_loss_ratio=%f\nno_vis_loss=%r\nmax_sample_frame=%d\nlr=%f\nweight_decay=%f\nbatch_size=%d\n\n' % 
            (use_ecc, use_modulator, use_bn, use_residual, vis_roi_features, no_visrepr, vis_loss_ratio, no_vis_loss, max_sample_frame, lr, weight_decay, batch_size))
        f.write('[Loss log]\n')

    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    #################
    # Load Datasets #
    #################
    train_set = MOT17SimpleReIDWrapper('train', 0.8, 0.0, max_sample_frame, tracker_cfg=tracker_config)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=simple_reid_wrapper_collate)
    val_set = MOT17SimpleReIDWrapper('val', 0.8, 0.0, max_sample_frame, tracker_cfg=tracker_config)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=simple_reid_wrapper_collate)

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

    if v2:
        motion_model = MotionModelSimpleReIDV2(use_modulator=use_modulator, use_bn=use_bn, use_residual=use_residual, 
                                               vis_roi_features=vis_roi_features, no_visrepr=no_visrepr)
    else:
        motion_model = MotionModelSimpleReID(use_modulator=use_modulator, use_bn=use_bn, use_residual=use_residual, 
                                             vis_roi_features=vis_roi_features, no_visrepr=no_visrepr)
    motion_model.train()
    motion_model.cuda()

    if use_refine_model:
        motion_model = RefineModel(motion_model)
        motion_model.train()
        motion_model.cuda()

    reid_network = resnet50(pretrained=False, output_dim=128)
    reid_network.load_state_dict(torch.load(tracker_config['tracktor']['reid_weights'],
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

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
            early_reid = get_batch_mean_early_reid(reid_network, data['early_reid_patches'])
            curr_reid = reid_network(data['curr_reid_patch'].cuda())

            conv_features, repr_features = get_features(obj_detect, data['curr_img'], data['curr_gt_app'])

            prev_loc = (data['prev_gt_warped'] if use_ecc else data['prev_gt']).cuda()
            curr_loc = (data['curr_gt_warped'] if use_ecc else data['curr_gt']).cuda()
            label_loc = data['label_gt'].cuda()
            curr_vis = data['curr_vis'].cuda()

            n_iter += 1
            optimizer.zero_grad()
            if use_refine_model:
                pred_loc_wh, vis = motion_model(obj_detect, data['label_img'], conv_features, repr_features, prev_loc, curr_loc,
                                                early_reid=early_reid, curr_reid=curr_reid)
                label_loc_wh = two_p_to_wh(label_loc)

                pred_loss = weighted_smooth_l1_loss(pred_loc_wh, label_loc_wh, curr_vis)
                vis_loss = vis_loss_func(vis, curr_vis)
            else:
                if v2:
                    pred_loc_wh, vis = motion_model(early_reid, curr_reid, repr_features, prev_loc, curr_loc)
                else:
                    pred_loc_wh, vis = motion_model(early_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc)
                label_loc_wh = two_p_to_wh(label_loc)

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
            for data in val_loader:
                early_reid = get_batch_mean_early_reid(reid_network, data['early_reid_patches'])
                curr_reid = reid_network(data['curr_reid_patch'].cuda())

                conv_features, repr_features = get_features(obj_detect, data['curr_img'], data['curr_gt_app'])

                prev_loc = (data['prev_gt_warped'] if use_ecc else data['prev_gt']).cuda()
                curr_loc = (data['curr_gt_warped'] if use_ecc else data['curr_gt']).cuda()
                label_loc = data['label_gt'].cuda()
                curr_vis = data['curr_vis'].cuda()

                if use_refine_model:
                    pred_loc_wh, vis = refine_model(obj_detect, data['label_img'], conv_features, repr_features, prev_loc, curr_loc,
                                                    early_reid=early_reid, curr_reid=curr_reid)
                    label_loc_wh = two_p_to_wh(label_loc)

                    pred_loss = weighted_smooth_l1_loss(pred_loc_wh, label_loc_wh, curr_vis)
                    vis_loss = vis_loss_func(vis, curr_vis)
                else:
                    if v2:
                        pred_loc_wh, vis = motion_model(early_reid, curr_reid, repr_features, prev_loc, curr_loc)
                    else:
                        pred_loc_wh, vis = motion_model(early_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc)
                    label_loc_wh = two_p_to_wh(label_loc)

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

        motion_model.train()

        if mean_val_pred_loss < lowest_val_loss:
            lowest_val_loss, lowest_val_loss_epoch = mean_val_pred_loss, epoch + 1
            last_save_epoch = lowest_val_loss_epoch
            new_lowest_flag = True
            torch.save(motion_model.state_dict(), osp.join(output_dir, 'simple_reid_motion_model_epoch_%d.pth'%(epoch+1)))
        elif epoch + 1 - last_save_epoch == save_epoch_freq:
            last_save_epoch = epoch + 1
            torch.save(motion_model.state_dict(), osp.join(output_dir, 'simple_reid_motion_model_epoch_%d.pth'%(epoch+1)))

        with open(log_file, 'a') as f:
            f.write('Epoch %4d: train pred loss %.6f, vis loss %.6f; val pred loss %.6f, vis loss %.6f %s\n' % 
                (epoch+1, mean_train_pred_loss, mean_train_vis_loss, mean_val_pred_loss, mean_val_vis_loss, '*' if new_lowest_flag else ''))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/cs/student/vbox/tianjliu/tracktor_output/motion_model')
    parser.add_argument('--pretrain_vis_path', type=str, default='/cs/student/vbox/tianjliu/tracktor_output/vis_model_epoch_94.pth')
    parser.add_argument('--ex_name', type=str, default='simple_default')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--use_ecc', action='store_true')
    parser.add_argument('--use_modulator', action='store_true')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--use_residual', action='store_true')
    parser.add_argument('--vis_roi_features', action='store_true')
    parser.add_argument('--no_visrepr', action='store_true')
    parser.add_argument('--vis_loss_ratio', type=float, default=1.0)
    parser.add_argument('--no_vis_loss', action='store_true')

    parser.add_argument('--max_sample_frame', type=int, default=2)
    parser.add_argument('--v2', action='store_true')
    parser.add_argument('--refine_model', action='store_true')

    args = parser.parse_args()
    print(args)

    train_main(args.v2, args.refine_model, args.use_ecc, args.use_modulator, args.use_bn, args.use_residual, args.vis_roi_features, 
               args.no_visrepr, args.vis_loss_ratio, args.no_vis_loss, args.max_sample_frame,
               args.lr, args.weight_decay, args.batch_size, args.output_dir, args.ex_name)