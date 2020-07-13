import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import encode_motion, decode_motion, two_p_to_wh, wh_to_two_p

class MotionModelSimpleReIDV2(nn.Module):
    def __init__(self, reid_dim=128, representation_dim=1024, motion_repr_dim=512, 
                 use_modulator=True, use_bn=False, use_residual=True, vis_roi_features=True, no_visrepr=False):
        super(MotionModelSimpleReIDV2, self).__init__()

        self.reid_dim = reid_dim
        self.representation_dim = representation_dim
        self.vis_repr_dim = representation_dim // 2

        self.use_modulator = use_modulator
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.vis_roi_features = vis_roi_features
        self.no_visrepr = no_visrepr

        self.activation = nn.ReLU()

        if use_bn:
            # reid/visibility branch #
            self.compare_reid = nn.Sequential(
                nn.Linear(2 * reid_dim, 2 * reid_dim),
                nn.BatchNorm1d(2 * reid_dim), self.activation
            )
            if vis_roi_features:
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
                if use_modulator:
                    self.vis_modulate = nn.Sequential(
                        nn.Linear(self.vis_repr_dim, 4),
                        nn.Sigmoid()
                    )
                else:
                    self.vis_modulate = nn.Sequential(
                        nn.Linear(self.vis_repr_dim, 1),
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
            # reid/visibility branch #
            self.compare_reid = nn.Sequential(
                nn.Linear(2 * reid_dim, 2 * reid_dim),
                self.activation
            )
            if vis_roi_features:
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
                if use_modulator:
                    self.vis_modulate = nn.Sequential(
                        nn.Linear(self.vis_repr_dim, 4),
                        nn.Sigmoid()
                    )
                else:
                    self.vis_modulate = nn.Sequential(
                        nn.Linear(self.vis_repr_dim, 1),
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

    def forward(self, early_reid, curr_reid, representation_feature, previous_loc, curr_loc, 
                output_motion=False, output_vis_feature=False):

        previous_loc_wh = two_p_to_wh(previous_loc)
        curr_loc_wh = two_p_to_wh(curr_loc)

        input_motion = encode_motion(previous_loc_wh, curr_loc_wh)

        # reid
        compared_reid_feature = self.compare_reid(torch.cat([early_reid, curr_reid], 1))

        if self.vis_roi_features:
            vis_feature = self.fuse_reid_roi(torch.cat([compared_reid_feature, representation_feature], 1))
        else:
            vis_feature = self.fuse_reid_roi(compared_reid_feature)

        vis_output = torch.sigmoid(self.vis_out(vis_feature))

        if self.use_residual:
            modulator = self.vis_modulate(vis_feature)

        # motion
        motion_repr_feature = self.motion_repr(input_motion)

        # motion residual prediction
        if self.no_visrepr:
            motion_residual = self.motion_regress(
                torch.cat([representation_feature, motion_repr_feature], 1)
            )
        else:
            motion_residual = self.motion_regress(
                torch.cat([representation_feature, motion_repr_feature, vis_feature], 1)
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