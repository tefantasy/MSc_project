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
from tracktor.reid.resnet import resnet50

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
    batch_reid_features = []
    for reid_patch in early_reid_patches:
        reid_features = reid_model(reid_patch.cuda())
        reid_features = torch.mean(reid_features, 0)
        batch_reid_features.append(reid_features)
    batch_reid_features = torch.stack(batch_reid_features, 0)
    return batch_reid_features



def train_main(sgd, lr, weight_decay, batch_size, output_dir, ex_name):
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
        f.write('lr=%f\nweight_decay=%f\nbatch_size=%d\n\n' % 
            (lr, weight_decay, batch_size))
        f.write('[Loss log]\n')

    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    #################
    # Load Datasets #
    #################
    train_set = MOT17SimpleReIDWrapper('train', 0.8, 0.0, 1, train_random_sample=False, ecc=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=simple_reid_wrapper_collate)
    val_set = MOT17SimpleReIDWrapper('val', 0.8, 0.0, 1, train_random_sample=False, ecc=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=simple_reid_wrapper_collate)

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

    vis_model = VisSimpleReID()
    
    vis_model.train()
    vis_model.cuda()

    if sgd:
        optimizer = torch.optim.SGD(vis_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(vis_model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()

    #######################
    # Training Parameters #
    #######################
    max_epochs = 100
    log_freq = 25

    lowest_val_loss = 9999999.9
    lowest_val_loss_epoch = -1

    ############
    # Training #
    ############
    for epoch in range(max_epochs):
        n_iter = 0
        new_lowest_flag = False
        train_loss_iters = []
        val_loss_iters = []

        for data in train_loader:
            early_reid = get_batch_mean_early_reid(reid_network, data['early_reid_patches'])
            curr_reid = reid_network(data['curr_reid_patch'].cuda())
            conv_features, repr_features = get_features(obj_detect, data['curr_img'], data['curr_gt_app'])

            curr_vis = data['curr_vis'].cuda()
            
            n_iter += 1
            optimizer.zero_grad()
            vis = vis_model(early_reid, curr_reid, conv_features, repr_features)
            loss = loss_func(vis, curr_vis)

            loss.backward()
            optimizer.step()

            train_loss_iters.append(loss.item())
            if n_iter % log_freq == 0:
                print('[Train Iter %5d] train loss %.6f ...' % 
                    (n_iter, np.mean(train_loss_iters[n_iter-log_freq:n_iter])),
                    flush=True)

        mean_train_loss = np.mean(train_loss_iters)
        print('Train epoch %4d end.' % (epoch + 1))

        vis_model.eval()

        with torch.no_grad():
            for data in val_loader:
                early_reid = get_batch_mean_early_reid(reid_network, data['early_reid_patches'])
                curr_reid = reid_network(data['curr_reid_patch'].cuda())
                conv_features, repr_features = get_features(obj_detect, data['curr_img'], data['curr_gt_app'])

                curr_vis = data['curr_vis'].cuda()

                vis = vis_model(early_reid, curr_reid, conv_features, repr_features)
                loss = loss_func(vis, curr_vis)

                val_loss_iters.append(loss.item())

        mean_val_loss = np.mean(val_loss_iters)
        print('[Epoch %4d] train loss %.6f, val loss %.6f' % 
               (epoch+1, mean_train_loss, mean_val_loss))

        vis_model.train()

        if mean_val_loss < lowest_val_loss:
            lowest_val_loss, lowest_val_loss_epoch = mean_val_loss, epoch + 1
            new_lowest_flag = True
            torch.save(vis_model.state_dict(), osp.join(output_dir, 'vis_model_epoch_%d.pth'%(epoch+1)))

        with open(log_file, 'a') as f:
            f.write('[Epoch %4d] train loss %.6f, val loss %.6f %s\n' % 
                    (epoch+1, mean_train_loss, mean_val_loss, '*' if new_lowest_flag else ''))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/cs/student/vbox/tianjliu/tracktor_output')
    parser.add_argument('--ex_name', type=str, default='visreid_default')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sgd', action='store_true')

    args = parser.parse_args()
    print(args)

    train_main(args.sgd, args.lr, args.weight_decay, args.batch_size, args.output_dir, args.ex_name)