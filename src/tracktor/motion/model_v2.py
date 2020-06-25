import torch
import torch.nn as nn

from .utils import encode_motion, decode_motion, two_p_to_wh, wh_to_two_p

class MotionModelV2(nn.Module):
    def __init__(self, output_dim=256, pool_size=7, representation_dim=1024, motion_repr_dim=512, 
                 vis_conv_only=True, use_modulator=True, use_bn=False):
        super(MotionModelV2, self).__init__()

        self.output_dim = output_dim
        self.pool_size = pool_size
        self.representation_dim = representation_dim
        self.vis_repr_dim = representation_dim

        self.use_modulator = use_modulator
        self.use_bn = use_bn

        self.activation = nn.ReLU()

        if use_bn:
            # appearance branch #
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
            self.appearance_fuse = nn.Sequential(
                nn.Linear(2 * representation_dim, representation_dim),
                nn.BatchNorm1d(representation_dim), self.activation
            )
            # visibility branch #
            self.vis_conv = nn.Sequential(
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
            self.vis_fuse = nn.Sequential(
                nn.Linear(2 * representation_dim, self.vis_repr_dim),
                nn.BatchNorm1d(self.vis_repr_dim), self.activation
            )
            self.vis_out = nn.Linear(self.vis_repr_dim, 1)
            if use_modulator:
                self.vis_modulate = nn.Sequential(
                    nn.Linear(self.vis_repr_dim, 4),
                    nn.Sigmoid()
                )
            # motion branch #
            self.motion_repr = nn.Sequential(
                nn.Linear(4, motion_repr_dim),
                nn.BatchNorm1d(motion_repr_dim), self.activation
            )
            # motion residual regression #
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
                nn.Conv2d(output_dim, output_dim * 2, 3),
                self.activation,
                nn.Conv2d(output_dim * 2, output_dim * 2, 3),
                self.activation,
                nn.Conv2d(output_dim * 2, representation_dim, 3),
                self.activation
            )
            self.appearance_fuse = nn.Sequential(
                nn.Linear(2 * representation_dim, representation_dim),
                self.activation
            )
            # visibility branch #
            self.vis_conv = nn.Sequential(
                nn.Conv2d(output_dim, output_dim * 2, 3),
                self.activation,
                nn.Conv2d(output_dim * 2, output_dim * 2, 3),
                self.activation,
                nn.Conv2d(output_dim * 2, representation_dim, 3),
                self.activation
            )
            self.vis_fuse = nn.Sequential(
                nn.Linear(2 * representation_dim, self.vis_repr_dim),
                self.activation
            )
            self.vis_out = nn.Linear(self.vis_repr_dim, 1)
            if use_modulator:
                self.vis_modulate = nn.Sequential(
                    nn.Linear(self.vis_repr_dim, 4),
                    nn.Sigmoid()
                )
            # motion branch #
            self.motion_repr = nn.Sequential(
                nn.Linear(4, motion_repr_dim),
                self.activation
            )
            # motion residual regression #
            self.motion_regress = nn.Sequential(
                nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, representation_dim + motion_repr_dim + self.vis_repr_dim),
                self.activation,
                nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, motion_repr_dim),
                self.activation,
                nn.Linear(motion_repr_dim, 4)
            )

    def forward(self, roi_pool_output, representation_feature, previous_loc, curr_loc, output_motion=False):
        """
        Input and output bboxs (locations) are represented by (x1, y1, x2, y2) coordinates.

        If using ECC, then curr_loc and previous_loc should be positions WARPED to label_img.
        We don't check whether ECC is used here. 
        """
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