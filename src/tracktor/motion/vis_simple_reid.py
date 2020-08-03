import torch
import torch.nn as nn
import torch.nn.functional as F

class VisSimpleReID(nn.Module):
    def __init__(self, reid_dim=128, roi_output_dim=256, pool_size=7, representation_dim=1024):
        super(VisSimpleReID, self).__init__()

        self.reid_dim = reid_dim
        self.roi_output_dim = roi_output_dim
        self.pool_size = pool_size
        self.representation_dim = representation_dim
        self.vis_repr_dim = representation_dim // 2

        self.activation = nn.ReLU()



        self.compare_reid = nn.Sequential(
            nn.Linear(2 * reid_dim, 2 * reid_dim),
            self.activation
        )
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

        self.vis_out = nn.Linear(self.vis_repr_dim, 1)


    def forward(self, early_reid, curr_reid, roi_pool_output, representation_feature, output_feature=False):
        compared_reid_feature = self.compare_reid(torch.cat([early_reid, curr_reid], 1))

        vis_spatial_feature = self.vis_conv(roi_pool_output).squeeze(-1).squeeze(-1)
        vis_feature = self.vis_fuse(torch.cat([representation_feature, vis_spatial_feature], 1))
        vis_feature = self.fuse_reid_roi(torch.cat([compared_reid_feature, vis_feature], 1))

        vis_output = torch.sigmoid(self.vis_out(vis_feature))

        if output_feature:
            return vis_output.squeeze(-1), vis_feature
        else:
            return vis_output.squeeze(-1)