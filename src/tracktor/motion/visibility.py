import torch
import torch.nn as nn
import torch.nn.functional as F


class VisEst(nn.Module):
    def __init__(self, output_dim=256, pool_size=7, representation_dim=1024):
        super(VisEst, self).__init__()
        self.output_dim = output_dim
        self.pool_size = pool_size
        self.representation_dim = representation_dim

        self.activation = nn.ReLU()
        self.bn_fc = nn.BatchNorm1d(representation_dim)
        
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

        # self.fc_fuse = nn.Linear(2 * representation_dim, representation_dim)
        self.fc_fuse = nn.Linear(representation_dim, representation_dim)
        self.fc_vis = nn.Linear(representation_dim, 1)
        self.fc_representation = nn.Linear(representation_dim, representation_dim // 2)


    def forward(self, roi_pool_output, representation_feature):
        spatial_feature = self.vis_conv(roi_pool_output).squeeze(-1).squeeze(-1)
        # fused_feature = torch.cat([representation_feature, spatial_feature], 1)
        # fused_feature = self.fc_fuse(fused_feature)
        fused_feature = self.fc_fuse(spatial_feature)
        fused_feature = self.activation(self.bn_fc(fused_feature))
        
        vis = self.fc_vis(fused_feature)
        vis = torch.sigmoid(vis)
        vis_representation = self.fc_representation(fused_feature)

        return vis, vis_representation
