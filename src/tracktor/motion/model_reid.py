import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import encode_motion, decode_motion, two_p_to_wh, wh_to_two_p

class MotionModelReID(nn.Module):
    def __init__(self, reid_dim=128, roi_output_dim=256, pool_size=7, representation_dim=1024, motion_repr_dim=512, 
                 use_modulator=True, use_bn=False, use_residual=True, use_reid_distance=True):
        super(MotionModelReID, self).__init__()

        self.reid_dim = reid_dim
        self.roi_output_dim = roi_output_dim
        self.pool_size = pool_size
        self.representation_dim = representation_dim
        self.vis_repr_dim = representation_dim // 2

        self.use_modulator = use_modulator
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_reid_distance = use_reid_distance

        self.activation = nn.ReLU()

        # parameters and sub-modules
        self.temporal_reid_weight = nn.Parameter(torch.Tensor(reid_dim))

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
                nn.Linear(2 * representation_dim, self.vis_repr_dim),
                nn.BatchNorm1d(self.vis_repr_dim), self.activation
            )
            if not use_reid_distance:
                self.compare_reid = nn.Sequential(
                    nn.Linear(2 * reid_dim, reid_dim),
                    nn.BatchNorm1d(reid_dim), self.activation
                )

            self.fuse_reid_roi = nn.Sequential(
                nn.Linear(reid_dim + self.vis_repr_dim, self.vis_repr_dim),
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
            self.vis_conv = nn.Sequential(
                nn.Conv2d(roi_output_dim, roi_output_dim * 2, 3),
                self.activation,
                nn.Conv2d(roi_output_dim * 2, roi_output_dim * 2, 3),
                self.activation,
                nn.Conv2d(roi_output_dim * 2, representation_dim, 3),
                self.activation
            )
            self.vis_fuse = nn.Sequential(
                nn.Linear(2 * representation_dim, self.vis_repr_dim),
                self.activation
            )
            if not use_reid_distance:
                self.compare_reid = nn.Sequential(
                    nn.Linear(2 * reid_dim, reid_dim),
                    self.activation
                )

            self.fuse_reid_roi = nn.Sequential(
                nn.Linear(reid_dim + self.vis_repr_dim, self.vis_repr_dim),
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
            self.motion_regress = nn.Sequential(
                nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, representation_dim + motion_repr_dim + self.vis_repr_dim),
                self.activation,
                nn.Linear(representation_dim + motion_repr_dim + self.vis_repr_dim, motion_repr_dim),
                self.activation,
                nn.Linear(motion_repr_dim, 4)
            )

    def forward(self, historical_reid, curr_reid, roi_pool_output, representation_feature, previous_loc, curr_loc, output_motion=False):
        """
        Input and output bboxs (locations) are represented by (x1, y1, x2, y2) coordinates.

        Input:
            -historical_reid: list (len=batch) of (n_history, reid_dim)
            -curr_reid: (batch, reid_dim)
            -roi_pool_output: (batch, roi_output_dim, pool_size, pool_size)
            -representation_feature: (batch, representation_dim)
            -*loc: (batch, 4)

        If using ECC, then curr_loc and previous_loc should be positions WARPED to label_img.
        We don't check whether ECC is used here. 
        """

        previous_loc_wh = two_p_to_wh(previous_loc)
        curr_loc_wh = two_p_to_wh(curr_loc)

        input_motion = encode_motion(previous_loc_wh, curr_loc_wh)


        # get weighted historical reid features (temporal attention)
        batch_weighted_reid = []
        for historical_reid_feature in historical_reid:
            weighted_historical_feature = historical_reid_feature * self.temporal_reid_weight.expand(historical_reid_feature.size()[0], -1)
            dotted_feature = torch.sum(weighted_historical_feature, dim=1)
            coeff = F.softmax(dotted_feature, dim=0).unsqueeze(-1)
            weighted_feature = torch.sum(coeff * historical_reid_feature, dim=0)
            batch_weighted_reid.append(weighted_feature)
        reid_feature = torch.stack(batch_weighted_reid, 0) # (batch, reid_dim)

        # reid/visibility #
        if self.use_reid_distance:
            compared_reid_feature = (reid_feature - curr_reid) ** 2
        else:
            compared_reid_feature = self.compare_reid(torch.cat([reid_feature, curr_reid], 1))

        vis_spatial_feature = self.vis_conv(roi_pool_output).squeeze(-1).squeeze(-1)
        vis_feature = self.vis_fuse(torch.cat([representation_feature, vis_spatial_feature], 1))

        vis_feature = self.fuse_reid_roi(torch.cat([compared_reid_feature, vis_feature], 1))

        vis_output = torch.sigmoid(self.vis_out(vis_feature))
        if self.use_residual:
            modulator = self.vis_modulate(vis_feature)

        # appearance #
        spatial_feature = self.appearance_conv(roi_pool_output).squeeze(-1).squeeze(-1)
        appearance_feature = self.appearance_fuse(torch.cat([representation_feature, spatial_feature], 1))

        # motion #
        motion_repr_feature = self.motion_repr(input_motion)

        # motion residual prediction #
        motion_residual = self.motion_regress(
            torch.cat([appearance_feature, motion_repr_feature, vis_feature], 1)
        )

        # output #
        if self.use_residual:
            pred_motion = motion_residual * modulator + input_motion
        else:
            pred_motion = motion_residual

        pred_loc_wh = decode_motion(pred_motion, curr_loc_wh)

        if output_motion:
            return pred_motion
        else:
            return pred_loc_wh, vis_output.squeeze(-1)
