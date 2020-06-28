import torch
import torch.nn as nn

from .visibility import VisEst
from .utils import encode_motion, decode_motion, two_p_to_wh, wh_to_two_p
from .model import MotionModel
from .model_v2 import MotionModelV2

from tracktor.frcnn_fpn import FRCNN_FPN

class BackboneMotionModel(MotionModelV2):
    def __init__(self, tracker_config, output_dim=256, pool_size=7, representation_dim=1024, motion_repr_dim=512, 
                 vis_conv_only=True, use_modulator=True, use_bn=False):
        super().__init__(output_dim, pool_size, representation_dim, motion_repr_dim, vis_conv_only, use_modulator, use_bn)

        obj_detect = FRCNN_FPN(num_classes=2)
        obj_detect.load_state_dict(torch.load(tracker_config['tracktor']['obj_detect_model'],
                                   map_location=lambda storage, loc: storage))

        self.transform = obj_detect.transform
        self.backbone = obj_detect.backbone
        self.box_roi_pool = obj_detect.roi_heads.box_roi_pool
        self.box_head = obj_detect.roi_heads.box_head

    def forward(self, images, targets, previous_loc, curr_loc, output_motion=False):
        """
        images: list of image tensors (c, w, h), can be of different shapes.
        targets: for getting roi features. list of dicts, where the mandatory fields of the dicts are:
                 -"boxes": coordinate of boxes in (x1, y1, x2, y2) to be roi-pooled, shape (1, 4). 
        *_loc: for calculating motion.
        """

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # chage the structure of targets
        targets = [d['boxes'] for d in targets]

        roi_pool_output = self.box_roi_pool(features, targets, images.image_sizes)
        representation_feature = self.box_head(roi_pool_output)

        # backbone feature extraction ends

        previous_loc_wh = two_p_to_wh(previous_loc)
        curr_loc_wh = two_p_to_wh(curr_loc)

        input_motion = encode_motion(previous_loc_wh, curr_loc_wh)

        # appearance #
        spatial_feature = self.appearance_conv(roi_pool_output).squeeze(-1).squeeze(-1)
        appearance_feature = self.appearance_fuse(torch.cat([representation_feature, spatial_feature], 1))

        # visibility #
        vis_spatial_feature = self.vis_conv(roi_pool_output).squeeze(-1).squeeze(-1)
        vis_feature = self.vis_fuse(torch.cat([representation_feature, vis_spatial_feature], 1))

        vis_out = torch.sigmoid(self.vis_out(vis_feature))
        if self.use_modulator:
            modulator = self.vis_modulate(vis_feature)
        else:
            modulator = vis_out

        # motion #
        motion_repr_feature = self.motion_repr(input_motion)

        # motion residual prediction #
        motion_residual = self.motion_regress(
            torch.cat([appearance_feature, motion_repr_feature, vis_feature], 1)
        )

        # output #
        pred_motion = motion_residual * modulator + input_motion

        pred_loc_wh = decode_motion(pred_motion, curr_loc_wh)

        if output_motion:
            return pred_motion
        else:
            return pred_loc_wh, vis_out.squeeze(-1)

