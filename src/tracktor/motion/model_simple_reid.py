import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import encode_motion, decode_motion, two_p_to_wh, wh_to_two_p

class MotionModelSimpleReID(nn.Module):
    def __init__(self, reid_dim=128, roi_output_dim=256, pool_size=7, representation_dim=1024, motion_repr_dim=512, 
                 use_modulator=True, use_bn=False, use_residual=True, vis_roi_features=False, no_visrepr=False, modulate_from_vis=False):
        super(MotionModelSimpleReID, self).__init__()

        self.reid_dim = reid_dim
        self.roi_output_dim = roi_output_dim
        self.pool_size = pool_size
        self.representation_dim = representation_dim
        self.vis_repr_dim = representation_dim // 2

        self.use_modulator = use_modulator
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.vis_roi_features = vis_roi_features
        self.no_visrepr = no_visrepr
        self.modulate_from_vis = modulate_from_vis

        self.activation = nn.ReLU()

        if use_bn:
            # appearance branch #
            self.appearance_conv = nn.Sequential(
                nn.Conv2d(roi_output_dim, roi_output_dim * 2, 3),
                nn.BatchNorm2d(roi_output_dim * 2),
                self.activation,
                nn.Conv2d(roi_output_dim * 2, roi_output_dim * 2, 3),
                nn.BatchNorm2d(roi_output_dim * 2),
                self.activation,
                nn.Conv2d(roi_output_dim * 2, representation_dim, 3),
                nn.BatchNorm2d(representation_dim),
                self.activation
            )
            self.appearance_fuse = nn.Sequential(
                nn.Linear(2 * representation_dim, representation_dim),
                nn.BatchNorm1d(representation_dim), self.activation
            )
            # reid/visibility branch #
            self.compare_reid = nn.Sequential(
                nn.Linear(2 * reid_dim, 2 * reid_dim),
                nn.BatchNorm1d(2 * reid_dim), self.activation
            )
            if vis_roi_features:
                self.vis_conv = nn.Sequential(
                    nn.Conv2d(roi_output_dim, roi_output_dim * 2, 3),
                    nn.BatchNorm2d(roi_output_dim * 2),
                    self.activation,
                    nn.Conv2d(roi_output_dim * 2, roi_output_dim * 2, 3),
                    nn.BatchNorm2d(roi_output_dim * 2),
                    self.activation,
                    nn.Conv2d(roi_output_dim * 2, representation_dim, 3),
                    nn.BatchNorm2d(representation_dim),
                    self.activation
                )
                self.vis_fuse = nn.Sequential(
                    nn.Linear(2 * representation_dim, representation_dim),
                    nn.BatchNorm1d(representation_dim), self.activation
                )
                self.fuse_reid_roi = nn.Sequential(
                    nn.Linear(2 * reid_dim + representation_dim, self.vis_repr_dim),
                    nn.BatchNorm1d(self.vis_repr_dim), self.activation
                )
            else:
                self.fuse_reid_roi = nn.Sequential(
                    nn.Linear(2 * reid_dim, self.vis_repr_dim),
                    nn.BatchNorm1d(self.vis_repr_dim), self.activation
                )

            self.vis_out = nn.Linear(self.vis_repr_dim, 1)

            if use_residual:
                mod_in_channel = 1 if modulate_from_vis else self.vis_repr_dim
                mod_out_channel = 4 if use_modulator else 1
                
                self.vis_modulate = nn.Sequential(
                    nn.Linear(mod_in_channel, mod_out_channel),
                    nn.Sigmoid()
                )
            # motion branch #
            self.motion_repr = nn.Sequential(
                nn.Linear(4, motion_repr_dim),
                nn.BatchNorm1d(motion_repr_dim), self.activation
            )
            # motion residual regression #
            if no_visrepr:
                self.motion_regress = nn.Sequential(
                    nn.Linear(representation_dim + motion_repr_dim, representation_dim + motion_repr_dim),
                    nn.BatchNorm1d(representation_dim + motion_repr_dim),
                    self.activation,
                    nn.Linear(representation_dim + motion_repr_dim, motion_repr_dim),
                    nn.BatchNorm1d(motion_repr_dim),
                    self.activation,
                    nn.Linear(motion_repr_dim, 4)
                )
            else:
                self.motion_regress = nn.Sequential(
                    nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, representation_dim + motion_repr_dim + self.vis_repr_dim),
                    nn.BatchNorm1d(representation_dim + motion_repr_dim + self.vis_repr_dim),
                    self.activation,
                    nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, motion_repr_dim),
                    nn.BatchNorm1d(motion_repr_dim),
                    self.activation,
                    nn.Linear(motion_repr_dim, 4)
                )
        else:
            # appearance branch #
            self.appearance_conv = nn.Sequential(
                nn.Conv2d(roi_output_dim, roi_output_dim * 2, 3),
                self.activation,
                nn.Conv2d(roi_output_dim * 2, roi_output_dim * 2, 3),
                self.activation,
                nn.Conv2d(roi_output_dim * 2, representation_dim, 3),
                self.activation
            )
            self.appearance_fuse = nn.Sequential(
                nn.Linear(2 * representation_dim, representation_dim),
                self.activation
            )
            # reid/visibility branch #
            self.compare_reid = nn.Sequential(
                nn.Linear(2 * reid_dim, 2 * reid_dim),
                self.activation
            )
            if vis_roi_features:
                self.vis_conv = nn.Sequential(
                    nn.Conv2d(roi_output_dim, roi_output_dim * 2, 3),
                    self.activation,
                    nn.Conv2d(roi_output_dim * 2, roi_output_dim * 2, 3),
                    self.activation,
                    nn.Conv2d(roi_output_dim * 2, representation_dim, 3),
                    self.activation
                )
                self.vis_fuse = nn.Sequential(
                    nn.Linear(2 * representation_dim, representation_dim),
                    self.activation
                )
                self.fuse_reid_roi = nn.Sequential(
                    nn.Linear(2 * reid_dim + representation_dim, self.vis_repr_dim),
                    self.activation
                )
            else:
                self.fuse_reid_roi = nn.Sequential(
                    nn.Linear(2 * reid_dim, self.vis_repr_dim),
                    self.activation
                )

            self.vis_out = nn.Linear(self.vis_repr_dim, 1)

            if use_residual:
                mod_in_channel = 1 if modulate_from_vis else self.vis_repr_dim
                mod_out_channel = 4 if use_modulator else 1
                
                self.vis_modulate = nn.Sequential(
                    nn.Linear(mod_in_channel, mod_out_channel),
                    nn.Sigmoid()
                )
            # motion branch #
            self.motion_repr = nn.Sequential(
                nn.Linear(4, motion_repr_dim),
                self.activation
            )
            # motion residual regression #
            if no_visrepr:
                self.motion_regress = nn.Sequential(
                    nn.Linear(representation_dim + motion_repr_dim, representation_dim + motion_repr_dim),
                    self.activation,
                    nn.Linear(representation_dim + motion_repr_dim, motion_repr_dim),
                    self.activation,
                    nn.Linear(motion_repr_dim, 4)
                )
            else:
                self.motion_regress = nn.Sequential(
                    nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, representation_dim + motion_repr_dim + self.vis_repr_dim),
                    self.activation,
                    nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, motion_repr_dim),
                    self.activation,
                    nn.Linear(motion_repr_dim, 4)
                )

    def forward(self, early_reid, curr_reid, roi_pool_output, representation_feature, previous_loc, curr_loc, 
                output_motion=False, output_vis_feature=False):

        previous_loc_wh = two_p_to_wh(previous_loc)
        curr_loc_wh = two_p_to_wh(curr_loc)

        input_motion = encode_motion(previous_loc_wh, curr_loc_wh)

        # reid
        compared_reid_feature = self.compare_reid(torch.cat([early_reid, curr_reid], 1))

        if self.vis_roi_features:
            vis_spatial_feature = self.vis_conv(roi_pool_output).squeeze(-1).squeeze(-1)
            vis_feature = self.vis_fuse(torch.cat([representation_feature, vis_spatial_feature], 1))
            vis_feature = self.fuse_reid_roi(torch.cat([compared_reid_feature, vis_feature], 1))
        else:
            vis_feature = self.fuse_reid_roi(compared_reid_feature)

        vis_output = torch.sigmoid(self.vis_out(vis_feature))

        if self.use_residual:
            if self.modulate_from_vis:
                modulator = self.vis_modulate(vis_output)
            else:
                modulator = self.vis_modulate(vis_feature)

        # appearance
        spatial_feature = self.appearance_conv(roi_pool_output).squeeze(-1).squeeze(-1)
        appearance_feature = self.appearance_fuse(torch.cat([representation_feature, spatial_feature], 1))

        # motion
        motion_repr_feature = self.motion_repr(input_motion)

        # motion residual prediction
        if self.no_visrepr:
            motion_residual = self.motion_regress(
                torch.cat([appearance_feature, motion_repr_feature], 1)
            )
        else:
            motion_residual = self.motion_regress(
                torch.cat([appearance_feature, motion_repr_feature, vis_feature], 1)
            )

        if self.use_residual:
            pred_motion = motion_residual * modulator + input_motion
        else:
            pred_motion = motion_residual

        pred_loc_wh = decode_motion(pred_motion, curr_loc_wh)

        if output_motion:
            return pred_motion
        else:
            if output_vis_feature:
                return pred_loc_wh, vis_output.squeeze(-1), vis_feature
            else:
                return pred_loc_wh, vis_output.squeeze(-1)
