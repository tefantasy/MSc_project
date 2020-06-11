import torch
import torch.nn as nn

from .visibility import VisEst
from .utils import encode_motion, decode_motion, two_p_to_wh, wh_to_two_p

class MotionModel(nn.Module):
    def __init__(self, output_dim=256, pool_size=7, representation_dim=1024, motion_repr_dim=512, vis_conv_only=True):
        super(MotionModel, self).__init__()

        self.output_dim = output_dim
        self.pool_size = pool_size
        self.representation_dim = representation_dim
        self.vis_repr_dim = representation_dim // 2

        self.vis_module = VisEst(output_dim, pool_size, representation_dim, vis_conv_only)

        self.activation = nn.ReLU()

        self.appearance_conv = nn.Sequential(
            nn.Conv2d(output_dim, output_dim * 2, 3),
            nn.BatchNorm2d(output_dim * 2),
            self.activation,
            nn.Conv2d(output_dim * 2, output_dim * 2, 3),
            nn.BatchNorm2d(output_dim * 2),
            self.activation,
            nn.Conv2d(output_dim * 2, representation_dim, 3),
            nn.BatchNorm2d(representation_dim),
            self.activation
        )
        self.appearance_fuse = nn.Linear(2 * representation_dim, representation_dim)
        self.motion_repr = nn.Linear(4, motion_repr_dim)
        self.motion_regress = nn.Sequential(
            nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, representation_dim + motion_repr_dim + self.vis_repr_dim)
            nn.BatchNorm1d(representation_dim + motion_repr_dim + self.vis_repr_dim),
            self.activation,
            nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, motion_repr_dim),
            nn.BatchNorm1d(motion_repr_dim),
            self.activation,
            nn.Linear(motion_repr_dim, 4)
        )
        self.modulate = nn.Linear(1, 4)

        self.bn_appearance = nn.BatchNorm1d(representation_dim)
        self.bn_motion_input = nn.BatchNorm1d(motion_repr_dim)
        self.bn_vis_repr = nn.BatchNorm1d(self.vis_repr_dim)
        self.bn_modulate = nn.BatchNorm1d(4)

    def load_vis_pretrained(self, weight_path):
        self.vis_module.load_state_dict(torch.load(weight_path))

    def forward(self, roi_pool_output, representation_feature, previous_loc, curr_loc, curr_loc_warped=None):
        """
        Input and output bboxs (locations) are represented by (x1, y1, x2, y2) coordinates.

        If using ECC, then curr_loc_warped must be provided, and previous_loc should be WARPED previous locations.
        """
        previous_loc_wh = two_p_to_wh(previous_loc)
        curr_loc_wh = two_p_to_wh(curr_loc)

        input_motion = encode_motion(previous_loc_wh, curr_loc_wh)

        # main part of the module begins

        motion_repr_feature = self.motion_repr(input_motion)
        motion_repr_feature = self.activation(self.bn_motion_input(motion_repr_feature))

        spatial_feature = self.appearance_conv(roi_pool_output).squeeze(-1).squeeze(-1)
        appearance_feature = self.appearance_fuse(torch.cat([representation_feature, spatial_feature], 1))
        appearance_feature = self.activation(self.bn_appearance(appearance_feature))

        vis, vis_repr_feature = self.vis_module(roi_pool_output, representation_feature)
        vis_repr_feature = self.activation(self.bn_vis_repr(vis_repr_feature))

        motion_residual = self.motion_regress(
            torch.cat([appearance_feature, motion_repr_feature, vis_repr_feature], 1))

        modulator = torch.sigmoid(self.bn_modulate(self.modulate(vis)))

        pred_motion = motion_residual * modulator + input_motion

        # main part of the module ends

        if curr_loc_warped is None:
            pred_loc_wh = decode_motion(pred_motion, curr_loc_wh)
        else:
            # use ECC
            curr_loc_warped_wh = two_p_to_wh(curr_loc_warped)
            pred_loc_wh = decode_motion(pred_motion, curr_loc_warped_wh)
        pred_loc = wh_to_two_p(pred_loc_wh)

        return pred_loc, vis.squeeze(-1)

