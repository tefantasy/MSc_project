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

from tracktor.config import cfg, get_output_dir
from tracktor.datasets.mot17_clips_wrapper import MOT17ClipsWrapper, clips_wrapper_collate

from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.motion.model_reid import MotionModelReID
from tracktor.reid.resnet import resnet50

from tracktor.motion.utils import two_p_to_wh



class BatchForger(object):
    """
    Deal with batches with variant sizes.
    """
    def __init__(self, batch_size, sample_shape):
        self.batch_size = batch_size
        self.sample_shape = sample_shape

        self.buffer = torch.zeros(batch_size * 10, *sample_shape, dtype=torch.float32).cuda()

        self.num_samples = 0

    def feed(self, data):
        """
        data: (size, *sample_shape)
        """
        with torch.no_grad():
            num_data = data.size()[0]
            assert self.num_samples+num_data < self.batch_size*10
            self.buffer[self.num_samples:self.num_samples+num_data] = data
            self.num_samples += num_data

    def dump(self):
        if not self.has_one_batch():
            return torch.zeros(0, *sample_shape, dtype=torch.float32).cuda()
        with torch.no_grad():
            batch_data = self.buffer[:self.batch_size].clone()
            new_buffer_data = self.buffer[self.batch_size:self.num_samples].clone()
            self.num_samples -= self.batch_size
            self.buffer[:self.num_samples] = new_buffer_data
        return batch_data

    def reset(self):
        self.num_samples = 0

    def has_one_batch(self):
        return self.num_samples >= self.batch_size

class BatchForgerList(object):
    """
    Deal with batches with variant sizes. Here data are stored in lists instead of Tensors.
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.buffer = []
    def feed(self, data):
        self.buffer.extend(data)
    def dump(self):
        if not self.has_one_batch():
            return []
        buffer_temp = self.buffer
        self.buffer = self.buffer[self.batch_size:]
        return buffer_temp[:self.batch_size]
    def reset(self):
        self.buffer = []
    def has_one_batch(self):
        return len(self.buffer) >= self.batch_size

class BatchForgerManager(object):
    def __init__(self, batch_forger_list):
        """
        All the batch forgers should have the same batch sizes!
        """
        self.batch_forger_list = batch_forger_list
        self.num_batch_forger = len(self.batch_forger_list)
        assert self.num_batch_forger > 0
    def feed(self, data_tuple):
        assert len(data_tuple) == self.num_batch_forger
        for i, data in enumerate(data_tuple):
            self.batch_forger_list[i].feed(data)
    def dump(self):
        output_data_list = []
        for batch_forger in self.batch_forger_list:
            output_data_list.append(batch_forger.dump())
        return tuple(output_data_list)
    def reset(self):
        for batch_forger in self.batch_forger_list:
            batch_forger.reset()
    def has_one_batch(self):
        for batch_forger in self.batch_forger_list:
            if not batch_forger.has_one_batch():
                return False
        return True


def get_features(obj_detect, img_list, curr_frame_offset, curr_gt_app):
    """
    Input:
        -img_list: list (len=clip_len) of (3, w, h). Can be different sizes. 
        -curr_frame_offset: (batch,)
        -curr_gt_app: (batch, 4)
    Output:
        -box_features: (batch, 256, 7, 7) CUDA
        -box_head_features: (batch, 1024) CUDA
    """
    box_features_list = []
    box_head_features_list = []

    with torch.no_grad():
        gts = curr_gt_app.cuda()
        for i, frame_idx in enumerate(curr_frame_offset):
            obj_detect.load_image(img_list[frame_idx].unsqueeze(0))

            gt = gts[i].unsqueeze(0)
            gt = resize_boxes(gt, obj_detect.original_image_sizes[0], obj_detect.preprocessed_images.image_sizes[0])
            gt = [gt]

            box_features = obj_detect.roi_heads.box_roi_pool(obj_detect.features, gt, obj_detect.preprocessed_images.image_sizes)
            box_head_features = obj_detect.roi_heads.box_head(box_features)
            box_features_list.append(box_features.squeeze(0))
            box_head_features_list.append(box_head_features.squeeze(0))

        return torch.stack(box_features_list, 0), torch.stack(box_head_features_list, 0)


def get_batch_reid_features(reid_model, img_list, batch_history):
    trans = Compose([ToPILImage(), Resize((256,128)), ToTensor()])

    batch_features = []
    with torch.no_grad():
        for history in batch_history:
            frame_ind = history['frame_offset']
            gts = history['gt']
            imgs = [img_list[idx] for idx in frame_ind]

            img_patches = []
            for i in range(len(frame_ind)):
                img, pos = imgs[i], gts[i]

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
                img = img[:, y0:y1, x0:x1]
                img = trans(img)
                img_patches.append(img)
            img_patches = torch.stack(img_patches, 0).cuda()

            reid_features = reid_model(img_patches)
            batch_features.append(reid_features)
    return batch_features

def get_curr_reid_features(reid_model, img_list, curr_frame_offset, curr_gt_app):
    trans = Compose([ToPILImage(), Resize((256,128)), ToTensor()])

    img_patches = []
    with torch.no_grad():
        for i, frame_idx in enumerate(curr_frame_offset):
            img = img_list[frame_idx]
            pos = curr_gt_app[i]

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
            img = img[:, y0:y1, x0:x1]
            img = trans(img)
            img_patches.append(img)
        img_patches = torch.stack(img_patches, 0).cuda()

        reid_features = reid_model(img_patches)
    return reid_features



def train_main(use_ecc, use_modulator, use_bn, use_residual, use_reid_distance, vis_loss_ratio, no_vis_loss,
               lr, weight_decay, batch_size, output_dir, ex_name):
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
        f.write('use_ecc=%r\nuse_modulator=%r\nuse_bn=%r\nuse_residual=%r\nuse_reid_distance=%r\nvis_loss_ratio=%f\nno_vis_loss=%r\nlr=%f\nweight_decay=%f\nbatch_size=%d\n\n' % 
            (use_ecc, use_modulator, use_bn, use_residual, use_reid_distance, vis_loss_ratio, no_vis_loss, lr, weight_decay, batch_size))
        f.write('[Loss log]\n')

    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    #################
    # Load Datasets #
    #################
    train_set = MOT17ClipsWrapper('train', 0.8, 0.0, clip_len=10, train_jitter=True, ecc=True, tracker_cfg=tracker_config)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1, collate_fn=clips_wrapper_collate)
    val_set = MOT17ClipsWrapper('val', 0.8, 0.0, clip_len=10, train_jitter=True, ecc=True, tracker_cfg=tracker_config)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=clips_wrapper_collate)

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

    motion_model = MotionModelReID(use_modulator=use_modulator, use_bn=use_bn, use_residual=use_residual, use_reid_distance=use_reid_distance)

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

    # usage: historical_reid, curr_reid, roi_pool_output, representation_feature, prev_loc, curr_loc, curr_vis, label_loc
    batch_manager = BatchForgerManager([
        BatchForgerList(batch_size),
        BatchForger(batch_size, (motion_model.reid_dim,)),
        BatchForger(batch_size, (motion_model.roi_output_dim, motion_model.pool_size, motion_model.pool_size)),
        BatchForger(batch_size, (motion_model.representation_dim,)),
        BatchForger(batch_size, (4,)),
        BatchForger(batch_size, (4,)),
        BatchForger(batch_size, ()),
        BatchForger(batch_size, (4,))
    ])

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

        for data in train_loader:
            historical_reid = get_batch_reid_features(reid_network, data['imgs'], data['historical'])
            curr_reid = get_curr_reid_features(reid_network, data['imgs'], data['curr_frame_offset'], data['curr_gt_app'])
            conv_features, repr_features = get_features(obj_detect, data['imgs'], data['curr_frame_offset'], data['curr_gt_app'])
            prev_loc = (data['prev_gt_warped'] if use_ecc else data['prev_gt']).cuda()
            curr_loc = (data['curr_gt_warped'] if use_ecc else data['curr_gt']).cuda()
            curr_vis = data['curr_vis'].cuda()
            label_loc = data['label_gt'].cuda()

            batch_manager.feed((historical_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc, curr_vis, label_loc))

            while batch_manager.has_one_batch():
                n_iter += 1
                historical_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc, curr_vis, label_loc = \
                    batch_manager.dump()

                optimizer.zero_grad()

                pred_loc_wh, vis = motion_model(historical_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc)
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
                        (n_iter, np.mean(train_pred_loss_iters[n_iter-log_freq:n_iter]), 
                         np.mean(train_vis_loss_iters[n_iter-log_freq:n_iter])), flush=True)

        mean_train_pred_loss = np.mean(train_pred_loss_iters)
        mean_train_vis_loss = np.mean(train_vis_loss_iters)
        train_pred_loss_epochs.append(mean_train_pred_loss)
        train_vis_loss_epochs.append(mean_train_vis_loss)
        print('Train epoch %4d end.' % (epoch + 1))

        batch_manager.reset()
        motion_model.eval()

        with torch.no_grad():
            for data in val_loader:
                historical_reid = get_batch_reid_features(reid_network, data['imgs'], data['historical'])
                curr_reid = get_curr_reid_features(reid_network, data['imgs'], data['curr_frame_offset'], data['curr_gt_app'])
                conv_features, repr_features = get_features(obj_detect, data['imgs'], data['curr_frame_offset'], data['curr_gt_app'])
                prev_loc = (data['prev_gt_warped'] if use_ecc else data['prev_gt']).cuda()
                curr_loc = (data['curr_gt_warped'] if use_ecc else data['curr_gt']).cuda()
                curr_vis = data['curr_vis'].cuda()
                label_loc = data['label_gt'].cuda()

                batch_manager.feed((historical_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc, curr_vis, label_loc))

                while batch_manager.has_one_batch():
                    historical_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc, curr_vis, label_loc = \
                        batch_manager.dump()

                    pred_loc_wh, vis = motion_model(historical_reid, curr_reid, conv_features, repr_features, prev_loc, curr_loc)
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
            (epoch+1, mean_train_pred_loss, mean_train_vis_loss, mean_val_pred_loss, mean_val_vis_loss), flush=True)
        with open(log_file, 'a') as f:
            f.write('Epoch %4d: train pred loss %.6f, vis loss %.6f; val pred loss %.6f, vis loss %.6f\n' % 
                (epoch+1, mean_train_pred_loss, mean_train_vis_loss, mean_val_pred_loss, mean_val_vis_loss))

        batch_manager.reset()
        motion_model.train()

        if mean_val_pred_loss < lowest_val_loss:
            lowest_val_loss, lowest_val_loss_epoch = mean_val_pred_loss, epoch + 1
            torch.save(motion_model.state_dict(), osp.join(output_dir, 'reid_motion_model_epoch_%d.pth'%(epoch+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/cs/student/vbox/tianjliu/tracktor_output/motion_model')
    parser.add_argument('--ex_name', type=str, default='reid_motion_default')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--use_ecc', action='store_true')
    parser.add_argument('--use_modulator', action='store_true')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--use_residual', action='store_true')
    parser.add_argument('--use_reid_distance', action='store_true')
    parser.add_argument('--vis_loss_ratio', type=float, default=1.0)
    parser.add_argument('--no_vis_loss', action='store_true')

    args = parser.parse_args()
    print(args)

    train_main(args.use_ecc, args.use_modulator, args.use_bn, args.use_residual, args.use_reid_distance, 
               args.vis_loss_ratio, args.no_vis_loss,
               args.lr, args.weight_decay, args.batch_size, args.output_dir, args.ex_name)