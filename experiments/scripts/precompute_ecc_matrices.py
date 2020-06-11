import os.path as osp
import os
import pickle
import configparser

import torch
import numpy as np
import cv2
from PIL import Image

from torchvision.transforms import ToTensor

from tracktor.config import cfg


def ecc_align(im1, im2):
    """
    im1, im2: Tensor of shape (3, h, w)
    """
    im1 = np.transpose(im1.numpy(), (1, 2, 0))
    im2 = np.transpose(im2.numpy(), (1, 2, 0))
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                100, 0.00001)
    cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)

    return warp_matrix


mot17det_train_dir = osp.join(cfg.DATA_DIR, 'MOT17Det', 'train')
output_path = osp.join(cfg.ROOT_DIR, 'output', 'precomputed_ecc_matrices.pkl')

train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                 'MOT17-11', 'MOT17-13']


ecc_dict = {}

image_transform = ToTensor()

count = 0

for seq_name in train_folders:
    seq_path = osp.join(mot17det_train_dir, seq_name)
    
    config = configparser.ConfigParser()
    config.read(osp.join(seq_path,'seqinfo.ini'))

    seq_len = int(config['Sequence']['seqLength'])
    im_dir = config['Sequence']['imDir']
    im_dir = osp.join(seq_path, im_dir)

    # distance=1
    for i in range(1, seq_len):
        frame1, frame2 = i, i + 1
        im1_path = osp.join(im_dir, "{:06d}.jpg".format(frame1))
        im2_path = osp.join(im_dir, "{:06d}.jpg".format(frame2))

        im1 = image_transform(Image.open(im1_path).convert('RGB'))
        im2 = image_transform(Image.open(im2_path).convert('RGB'))

        warp_matrix = ecc_align(im1, im2)
        count += 1

        identifier = ','.join([seq_name, str(frame1), seq_name, str(frame2)])
        ecc_dict[identifier] = warp_matrix

        if count % 50 == 0:
            print(count)

    # distance=2
    for i in range(1, seq_len-1):
        frame1, frame2 = i, i + 2
        im1_path = osp.join(im_dir, "{:06d}.jpg".format(frame1))
        im2_path = osp.join(im_dir, "{:06d}.jpg".format(frame2))

        im1 = image_transform(Image.open(im1_path).convert('RGB'))
        im2 = image_transform(Image.open(im2_path).convert('RGB'))

        warp_matrix = ecc_align(im1, im2)
        count += 1

        identifier = ','.join([seq_name, str(frame1), seq_name, str(frame2)])
        ecc_dict[identifier] = warp_matrix

        if count % 50 == 0:
            print(count)

with open(output_path, 'wb') as f:
    pickle.dump(ecc_dict, f)