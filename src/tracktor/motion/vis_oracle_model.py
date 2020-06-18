import torch
import torch.nn as nn

from .visibility import VisEst
from .utils import encode_motion, decode_motion, two_p_to_wh, wh_to_two_p
from .model import MotionModel

class VisOracleMotionModel(MotionModel):
    def __init__(self, output_dim=256, pool_size=7, representation_dim=1024, motion_repr_dim=512, vis_conv_only=True, use_modulator=True):
        super().__init__(output_dim, pool_size, representation_dim, motion_repr_dim, vis_conv_only, use_modulator)

    def forward(self, roi_pool_output, representation_feature, previous_loc, curr_loc, vis_oracle):
        """
        vis_oracle: (batch, 1)
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

        pred_vis, vis_repr_feature = self.vis_module(roi_pool_output, representation_feature)
        vis_repr_feature = self.activation(self.bn_vis_repr(vis_repr_feature))

        motion_residual = self.motion_regress(
            torch.cat([appearance_feature, motion_repr_feature, vis_repr_feature], 1))

        if self.use_modulator:
            modulator = torch.sigmoid(self.bn_modulate(self.modulate(vis_oracle)))
        else:
            modulator = vis_oracle

        pred_motion = motion_residual * modulator + input_motion

        # main part of the module ends

        pred_loc_wh = decode_motion(pred_motion, curr_loc_wh)
        return pred_loc_wh, pred_vis.squeeze(-1)

