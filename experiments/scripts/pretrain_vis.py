import os.path as osp
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import resize_boxes

from tracktor.config import get_output_dir
from tracktor.datasets.mot17_vis import MOT17Vis

from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.motion.visibility import VisEst

class BatchForger(object):
    """
    Deal with batches with variant sizes.
    """
    def __init__(self, batch_size, sample_shape):
        self.batch_size = batch_size
        self.sample_shape = sample_shape

        self.batch = torch.zeros(batch_size, *sample_shape, dtype=torch.float32)
        self.residual = torch.zeros(batch_size * 10, *sample_shape, dtype=torch.float32)

        self.num_samples = 0

    def feed(self, data):
        """
        data: (size, *sample_shape)
        """
        with torch.no_grad():
            num_data = data.size()[0]
            if self.num_samples < self.batch_size:
                if num_data <= self.batch_size - self.num_samples:
                    self.batch[self.num_samples:self.num_samples+num_data] = data
                else:
                    self.batch[self.num_samples:] = data[:self.batch_size-self.num_samples]
                    self.residual[:num_data-(self.batch_size-self.num_samples)] = data[self.batch_size-self.num_samples:]
            else:
                self.residual[self.num_samples-self.batch_size:self.num_samples-self.batch_size+num_data] = data
            self.num_samples += num_data

    def dump(self):
        if not self.has_one_batch():
            return torch.zeros(0, *sample_shape, dtype=torch.float32)
        with torch.no_grad():
            batch_data = self.batch.clone()
            self.num_samples -= self.batch_size

            if self.num_samples <= self.batch_size:
                self.batch[:self.num_samples] = self.residual[:self.num_samples]
            else:
                self.batch[:] = self.residual[:self.batch_size]
                self.residual[:self.num_samples-self.batch_size] = self.residual.clone()[self.batch_size:self.num_samples]
        return batch_data

    def reset(self):
        self.num_samples = 0

    def has_one_batch(self):
        return self.num_samples >= self.batch_size

def get_features(obj_detect, img, gts):
    with torch.no_grad():
        obj_detect.load_image(img)

        gts = gts.squeeze(0).cuda()
        gts = resize_boxes(gts, obj_detect.original_image_sizes[0], obj_detect.preprocessed_images.image_sizes[0])
        gts = [gts]

        box_features = obj_detect.roi_heads.box_roi_pool(obj_detect.features, gts, obj_detect.preprocessed_images.image_sizes)
        box_head_features = obj_detect.roi_heads.box_head(box_features)

    return box_features.cpu(), box_head_features.cpu()

def pretrain_main():
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    np.random.seed(12345)
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir('motion'), 'vis')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    with open('experiments/cfgs/tracktor.yaml', 'r') as f:
        tracker_config = yaml.safe_load(f)

    #################
    # Load Datasets #
    #################
    train_set = MOT17Vis('train', 0.8, 0.0)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
    val_set = MOT17Vis('val', 0.8, 0.0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    ########################
    # Initializing Modules #
    ########################
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(tracker_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))
    obj_detect.eval()
    obj_detect.cuda()

    vis_model = VisEst()
    vis_model.train()
    vis_model.cuda()

    optimizer = torch.optim.Adam(vis_model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_func = nn.MSELoss()

    #######################
    # Training Parameters #
    #######################

    max_epochs = 100
    batch_size = 32

    conv_batch_forger = BatchForger(batch_size, (vis_model.output_dim, vis_model.pool_size, vis_model.pool_size))
    repr_batch_forger = BatchForger(batch_size, (vis_model.representation_dim,))
    label_batch_forger = BatchForger(batch_size, (1,))

    log_freq = 50

    train_loss_epochs = []
    val_loss_epochs = []
    lowest_val_loss = 9999999.9
    lowest_val_loss_epoch = -1

    ############
    # Training #
    ############

    for epoch in range(max_epochs):
        n_iter = 0
        train_loss_iters = []
        val_loss_iters = []


        for data in train_loader:
            conv_features, repr_features = get_features(obj_detect, data['img'], data['gt'])
            conv_batch_forger.feed(conv_features)
            repr_batch_forger.feed(repr_features)
            label_batch_forger.feed(data['vis'].squeeze(0).unsqueeze(-1))

            while label_batch_forger.has_one_batch():
                n_iter += 1
                batch_conv = conv_batch_forger.dump().cuda()
                batch_repr = repr_batch_forger.dump().cuda()
                batch_label = label_batch_forger.dump().cuda()

                optimizer.zero_grad()
                pred, _ = vis_model(batch_conv, batch_repr)
                loss = loss_func(pred, batch_label)
                loss.backward()
                optimizer.step()

                train_loss_iters.append(loss.item())
                if n_iter % log_freq == 0:
                    print('[Train Iter %5d] train loss %.6f ...' % (n_iter, np.mean(train_loss_iters[n_iter-log_freq:n_iter])))

        mean_train_loss = np.mean(train_loss_iters)
        train_loss_epochs.append(mean_train_loss)
        print('Train epoch %4d end.' % (epoch + 1))

        conv_batch_forger.reset()
        repr_batch_forger.reset()
        label_batch_forger.reset()

        vis_model.eval()

        for data in val_loader:
            conv_features, repr_features = get_features(obj_detect, data['img'], data['gt'])
            conv_batch_forger.feed(conv_features)
            repr_batch_forger.feed(repr_features)
            label_batch_forger.feed(data['vis'].squeeze(0).unsqueeze(-1))

            while label_batch_forger.has_one_batch():
                batch_conv = conv_batch_forger.dump().cuda()
                batch_repr = repr_batch_forger.dump().cuda()
                batch_label = label_batch_forger.dump().cuda()

                pred, _ = vis_model(batch_conv, batch_repr)
                loss = loss_func(pred, batch_label)

                val_loss_iters.append(loss.item())

        mean_val_loss = np.mean(val_loss_iters)
        val_loss_epochs.append(mean_val_loss)
        print('[Epoch %d] train loss %.6f, val loss %.6f' % (epoch+1, mean_train_loss, mean_val_loss))

        conv_batch_forger.reset()
        repr_batch_forger.reset()
        label_batch_forger.reset()

        vis_model.train()

        if mean_val_loss < lowest_val_loss:
            lowest_val_loss, lowest_val_loss_epoch = mean_val_loss, epoch + 1
            torch.save(vis_model.state_dict(), osp.join(output_dir, 'vis_model_epoch_%d.pth'%(epoch+1)))


pretrain_main()