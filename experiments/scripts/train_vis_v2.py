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
from tracktor.motion.vis_v2 import VisModel
from tracktor.datasets.mot17_vis_reid import MOT17VisReID, simple_reid_wrapper_collate

from tracktor.reid.resnet import resnet50

def get_batch_mean_early_reid(reid_model, early_reid_patches):
    with torch.no_grad():
        batch_reid_features = []
        for reid_patch in early_reid_patches:
            reid_features = reid_model(reid_patch.cuda())
            reid_features = torch.mean(reid_features, 0)
            batch_reid_features.append(reid_features)
        batch_reid_features = torch.stack(batch_reid_features, 0)
        return batch_reid_features

def weighted_mse_loss(pred, target, vis_gt):
    """
    pred: (batch, )
    target: (batch, )
    vis_gt: (batch, ) used to calculate weights
    """
    gamma = 4
    batch_size = pred.size()[0]

    loss_quadratic = (pred - target) ** 2
    weights = torch.pow(gamma, 1.0 - vis_gt)

    loss = torch.mean(loss_quadratic * weights) / 2.0

    return loss


def train_main(use_early_reid, use_reid_distance, lr, weight_decay, batch_size, output_dir, ex_name):
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
        f.write('use_early_reid=%r\nuse_reid_distance=%r\nlr=%f\nweight_decay=%f\nbatch_size=%d\n\n' % 
            (use_early_reid, use_reid_distance, lr, weight_decay, batch_size))
        f.write('[Loss log]\n')

    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    #################
    # Load Datasets #
    #################
    train_set = MOT17VisReID('train', 0.8, 0.0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=simple_reid_wrapper_collate)
    val_set = MOT17VisReID('val', 0.8, 0.0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=simple_reid_wrapper_collate)

    ########################
    # Initializing Modules #
    ########################
    reid_network = resnet50(pretrained=False, output_dim=128)
    reid_network.load_state_dict(torch.load(tracker_config['tracktor']['reid_weights'],
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    vis_model = VisModel(use_early_reid=use_early_reid, use_reid_distance=use_reid_distance)

    vis_model.train()
    vis_model.cuda()

    optimizer = torch.optim.SGD(vis_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    loss_func = nn.MSELoss()

    #######################
    # Training Parameters #
    #######################
    max_epochs = 100
    log_freq = 25

    lowest_val_loss = 9999999.9
    lowest_val_loss_epoch = -1
    lowest_val_weighed_loss = 9999999.9
    lowest_val_weighted_loss_epoch = -1

    for epoch in range(max_epochs):
        n_iter = 0
        new_lowest_flag = False
        train_loss_iters = []
        val_loss_iters = []
        val_weighted_loss_iters = []

        for data in train_loader:
            curr_patch = data['patches'].cuda()
            vis_gt = data['vis'].cuda()

            if use_early_reid:
                early_reid = get_batch_mean_early_reid(reid_network, data['early_patches'])
                curr_reid = reid_network(curr_patch)
            else:
                early_reid, curr_reid = None, None

            n_iter += 1
            optimizer.zero_grad()

            vis = vis_model(curr_patch, early_reid, curr_reid)
            loss = weighted_mse_loss(vis, vis_gt, vis_gt)
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
                curr_patch = data['patches'].cuda()
                vis_gt = data['vis'].cuda()

                if use_early_reid:
                    early_reid = get_batch_mean_early_reid(reid_network, data['early_patches'])
                    curr_reid = reid_network(curr_patch)
                else:
                    early_reid, curr_reid = None, None

                vis = vis_model(curr_patch, early_reid, curr_reid)
                loss = loss_func(vis, vis_gt)
                weighted_loss = weighted_mse_loss(vis, vis_gt, vis_gt)

                val_loss_iters.append(loss.item())
                val_weighted_loss_iters.append(weighted_loss.item())

        mean_val_loss = np.mean(val_loss_iters)
        mean_val_weighted_loss = np.mean(val_weighted_loss_iters)
        print('[Epoch %4d] train weighted loss %.6f, val weighted loss %.6f, val mse loss %.6f' % 
               (epoch+1, mean_train_loss, mean_val_weighted_loss, mean_val_loss))

        vis_model.train()

        if mean_val_weighted_loss < lowest_val_weighed_loss:
            lowest_val_weighed_loss, lowest_val_weighted_loss_epoch = mean_val_weighted_loss, epoch + 1
            lowest_val_loss = min(mean_val_loss, lowest_val_loss)
            new_lowest_flag = True
            torch.save(vis_model.state_dict(), osp.join(output_dir, 'vis_model_epoch_%d.pth'%(epoch+1)))
        elif mean_val_loss < lowest_val_loss:
            lowest_val_loss, lowest_val_loss_epoch = mean_val_loss, epoch + 1
            lowest_val_weighed_loss = min(mean_val_weighted_loss, lowest_val_weighed_loss)
            new_lowest_flag = True
            torch.save(vis_model.state_dict(), osp.join(output_dir, 'vis_model_epoch_%d.pth'%(epoch+1)))

        with open(log_file, 'a') as f:
            f.write('[Epoch %4d] train weighted loss %.6f, val weighted loss %.6f, val mse loss %.6f %s\n' % 
               (epoch+1, mean_train_loss, mean_val_weighted_loss, mean_val_loss, '*' if new_lowest_flag else ''))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/cs/student/vbox/tianjliu/tracktor_output')
    parser.add_argument('--ex_name', type=str, default='visv2_default')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--use_early_reid', action='store_true')
    parser.add_argument('--use_reid_distance', action='store_true')

    args = parser.parse_args()
    print(args)

    train_main(args.use_early_reid, args.use_reid_distance, args.lr, args.weight_decay, args.batch_size, args.output_dir, args.ex_name)

