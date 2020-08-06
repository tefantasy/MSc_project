import torch
import torch.nn as nn

from .vis_simple_reid import VisSimpleReID
from .visibility import VisEst
from .utils import encode_motion, decode_motion, two_p_to_wh, wh_to_two_p

class MotionModelV3(nn.Module):
    def __init__(self, vis_model, reid_dim=128, roi_output_dim=256, pool_size=7, representation_dim=1024, motion_repr_dim=512, correction=False, no_modulator=False, 
                 use_vis_model=True, use_motion_repr=True, use_vis_feature_for_mod=False, use_historical_appearance=False):
        super(MotionModelV3, self).__init__()

        self.use_vis_model = use_vis_model
        if use_vis_model:
            self.vis_model = vis_model
            assert isinstance(vis_model, VisSimpleReID) or isinstance(vis_model, VisEst)
            self.use_reid_vis_model = isinstance(vis_model, VisSimpleReID)

        self.use_motion_repr = use_motion_repr
        self.use_modulator = (not no_modulator)
        self.use_vis_feature_for_mod = use_vis_feature_for_mod
        self.use_historical_appearance = use_historical_appearance
        self.correction = correction

        self.vis_repr_dim = representation_dim // 2

        self.activation = nn.ReLU()

        # appearance branch #
        self.appearance_conv = nn.Sequential(
            nn.Conv2d(roi_output_dim, roi_output_dim * 2, 3),
            self.activation,
            nn.Conv2d(roi_output_dim * 2, roi_output_dim * 2, 3),
            self.activation,
            nn.Conv2d(roi_output_dim * 2, representation_dim, 3),
            self.activation
        )
        if use_historical_appearance:
            self.compare_reid = nn.Sequential(
                nn.Linear(2 * reid_dim, 2 * reid_dim),
                self.activation
            )
            self.appearance_fuse = nn.Sequential(
                nn.Linear(2 * representation_dim + 2 * reid_dim, representation_dim),
                self.activation
            )
        else:
            self.appearance_fuse = nn.Sequential(
                nn.Linear(2 * representation_dim, representation_dim),
                self.activation
            )

        # modulator #
        if use_vis_model:
            if use_vis_feature_for_mod:
                self.vis_modulate = nn.Sequential(
                    nn.Linear(self.vis_repr_dim, 4 if self.use_modulator else 1),
                    nn.Sigmoid()
                )
            elif self.use_modulator:
                self.vis_modulate = nn.Sequential(
                    nn.Linear(1, 4),
                    nn.Sigmoid()
                )
            # no module here if no_modulator and no vis_feature_for mod

        # motion branch #
        if use_motion_repr:
            self.motion_repr = nn.Sequential(
                nn.Linear(4, motion_repr_dim),
                self.activation
            )

        # motion residual regressor #
        if use_motion_repr:
            self.motion_regress = nn.Sequential(
                nn.Linear(representation_dim + motion_repr_dim, representation_dim + motion_repr_dim),
                self.activation,
                nn.Linear(representation_dim + motion_repr_dim, motion_repr_dim),
                self.activation,
                nn.Linear(motion_repr_dim, 4)
            )
        else:
            self.motion_regress = nn.Sequential(
                nn.Linear(representation_dim, representation_dim + motion_repr_dim),
                self.activation,
                nn.Linear(representation_dim + motion_repr_dim, motion_repr_dim),
                self.activation,
                nn.Linear(motion_repr_dim, 4)
            )

        # fix vis_model parameters
        for name, param in self.named_parameters():
            if name.startswith('vis_model'):
                param.requires_grad = False

    def get_trainable_parameters(self):
        """ only yield non-vis_model parameters (for training). """
        for name, param in self.named_parameters():
            if not name.startswith('vis_model'):
                yield param

    def forward(self, early_reid, curr_reid, roi_pool_output, representation_feature, previous_loc, curr_loc, 
                output_motion=False):
        previous_loc_wh = two_p_to_wh(previous_loc)
        curr_loc_wh = two_p_to_wh(curr_loc)

        input_motion = encode_motion(previous_loc_wh, curr_loc_wh)

        # appearance
        spatial_feature = self.appearance_conv(roi_pool_output).squeeze(-1).squeeze(-1)
        if self.use_historical_appearance:
            compared_reid_feature = self.compare_reid(torch.cat([early_reid, curr_reid], 1))
            appearance_feature = self.appearance_fuse(torch.cat([representation_feature, spatial_feature, compared_reid_feature], 1))
        else:
            appearance_feature = self.appearance_fuse(torch.cat([representation_feature, spatial_feature], 1))

        # motion
        if self.use_motion_repr:
            motion_repr_feature = self.motion_repr(input_motion)

        # vis and modulate
        if self.use_vis_model:
            if self.use_reid_vis_model:
                if self.use_vis_feature_for_mod:
                    vis_output, vis_feature = self.vis_model(early_reid, curr_reid, roi_pool_output, representation_feature, output_feature=True)
                    vis_output = vis_output.unsqueeze(-1)
                else:
                    vis_output = self.vis_model(early_reid, curr_reid, roi_pool_output, representation_feature).unsqueeze(-1)
            else:
                vis_output, _ = self.vis_model(roi_pool_output, representation_feature)

            if not self.use_vis_feature_for_mod and self.correction:
                vis_output = 15.0 * vis_output - 10.0
                vis_output = torch.sigmoid(vis_output)

            if self.use_vis_feature_for_mod:
                modulator = self.vis_modulate(vis_feature)
            elif self.use_modulator:
                modulator = self.vis_modulate(vis_output)
            else:
                modulator = vis_output

        # motion residual prediction
        if self.use_motion_repr:
            motion_residual = self.motion_regress(
                torch.cat([appearance_feature, motion_repr_feature], 1)
            )
        else:
            motion_residual = self.motion_regress(appearance_feature)

        # output
        if self.use_vis_model:
            pred_motion = motion_residual * modulator + input_motion
        else:
            pred_motion = motion_residual + input_motion

        if output_motion:
            return pred_motion
        else:
            pred_loc_wh = decode_motion(pred_motion, curr_loc_wh)
            if self.use_vis_model:
                return pred_loc_wh, vis_output.squeeze(-1)
            else:
                return pred_loc_wh, None