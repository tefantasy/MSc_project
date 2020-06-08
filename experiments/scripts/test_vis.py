import os.path as osp
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import resize_boxes

from tracktor.datasets.mot17_vis import MOT17Vis

from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.motion.visibility import VisEst


def get_features(obj_detect, img, gts):
    with torch.no_grad():
        obj_detect.load_image(img)

        gts = gts.squeeze(0).cuda()
        gts = resize_boxes(gts, obj_detect.original_image_sizes[0], obj_detect.preprocessed_images.image_sizes[0])
        gts = [gts]

        box_features = obj_detect.roi_heads.box_roi_pool(obj_detect.features, gts, obj_detect.preprocessed_images.image_sizes)
        box_head_features = obj_detect.roi_heads.box_head(box_features)

    return box_features.cpu(), box_head_features.cpu()

with open('experiments/cfgs/tracktor.yaml', 'r') as f:
    tracker_config = yaml.safe_load(f)

val_set = MOT17Vis('val', 0.8, 0.0, val_bbox_jitter=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=2)

obj_detect = FRCNN_FPN(num_classes=2)
obj_detect.load_state_dict(torch.load(tracker_config['tracktor']['obj_detect_model'],
                            map_location=lambda storage, loc: storage))
obj_detect.eval()
obj_detect.cuda()

vis_model = VisEst(conv_only=False)
vis_model.load_state_dict(torch.load(''))
vis_model.eval()
vis_model.cuda()

for data in val_loader:
    conv_features, repr_features = get_features(obj_detect, data['img'], data['gt'])
    conv_features = conv_features.cuda()
    repr_features = repr_features.cuda()
    label = data['vis'].squeeze(0).unsqueeze(-1).cuda()

    pred, _ = vis_model(conv_features, repr_features)

    print(label)
    print(pred)
    break