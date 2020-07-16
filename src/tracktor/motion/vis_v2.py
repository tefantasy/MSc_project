import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class VisModel(nn.Module):
    def __init__(self, use_early_reid=True, use_reid_distance=False):
        super(VisModel, self).__init__()

        self.use_early_reid = use_early_reid
        self.use_reid_distance = use_reid_distance

        backbone = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        
        if use_early_reid:
            if use_reid_distance:
                self.last_fc = nn.Linear(backbone.fc.in_features + 1, 1)
            else:
                self.fuse_reid = nn.Sequential(nn.Linear(2 * 128, 1), nn.ReLU())
                self.last_fc = nn.Linear(backbone.fc.in_features + 1, 1)
        else:
            self.last_fc = nn.Linear(backbone.fc.in_features, 1)

    def forward(self, curr_patch, early_reid=None, curr_reid=None):
        if self.use_early_reid:
            assert early_reid is not None and curr_reid is not None

        feature = self.backbone(curr_patch)
        feature = torch.flatten(feature, 1)

        if self.use_early_reid:
            if self.use_reid_distance:
                distance = torch.mean((early_reid - curr_reid) ** 2, 1, keepdim=True)
                feature = torch.cat([feature, distance], 1)
            else:
                fused_reid = self.fuse_reid(torch.cat([early_reid, curr_reid], 1))
                feature = torch.cat([feature, fused_reid], 1)

        vis = torch.sigmoid(self.last_fc(feature))

        return vis.squeeze(-1)