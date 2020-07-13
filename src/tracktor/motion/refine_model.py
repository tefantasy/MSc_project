import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import encode_motion, decode_motion, two_p_to_wh, wh_to_two_p

from .model import MotionModel
from .model_v2 import MotionModelV2
from .model_reid import MotionModelReID
from .model_simple_reid import MotionModelSimpleReID
from .model_simple_reid_v2 import MotionModelSimpleReIDV2

from torchvision.ops.boxes import clip_boxes_to_image
from torchvision.models.detection.transform import resize_boxes

class RefineModel(nn.Module):
    def __init__(self, motion_model, roi_output_dim=256, pool_size=7, representation_dim=1024):
        super(RefineModel, self).__init__()

        self.roi_output_dim = roi_output_dim
        self.pool_size = pool_size
        self.representation_dim = representation_dim

        self.motion_model = motion_model

        self.is_motion_model = isinstance(motion_model, MotionModel)
        self.is_motion_v2_model = isinstance(motion_model, MotionModelV2)
        self.is_reid_model = isinstance(motion_model, MotionModelReID)
        self.is_simple_reid_model = isinstance(motion_model, MotionModelSimpleReID)
        self.is_simple_reid_v2_model = isinstance(motion_model, MotionModelSimpleReIDV2)

        # refinement modules
        if self.is_motion_model:
            self.regress_modulator = nn.Sequential(nn.Linear(1, 4), nn.Sigmoid())
        else:
            self.regress_modulator = nn.Sequential(nn.Linear(self.motion_model.vis_repr_dim, 4), nn.Sigmoid())
        
        self.mlp_head = nn.Sequential(
            nn.Linear(roi_output_dim * pool_size ** 2, representation_dim), 
            nn.ReLU(),
            nn.Linear(representation_dim, representation_dim), 
            nn.ReLU()
        )
        self.regress_head = nn.Linear(representation_dim, 4)


    def get_roi_features(self, obj_detect, img_list, gts):
        """
        Input:
            -img_list: list of (1, 3, w, h). Can be different sizes. 
            -gts: (batch, 4)
        Output:
            -box_features: (batch, 256, 7, 7)
        """
        box_features_list = []

        with torch.no_grad():
            for i, img in enumerate(img_list):
                obj_detect.load_image(img)

                gt = gts[i].unsqueeze(0)
                gt = clip_boxes_to_image(gt, img.shape[-2:])
                gt = resize_boxes(gt, obj_detect.original_image_sizes[0], obj_detect.preprocessed_images.image_sizes[0])
                gt = [gt]

                box_features = obj_detect.roi_heads.box_roi_pool(obj_detect.features, gt, obj_detect.preprocessed_images.image_sizes)
                box_features_list.append(box_features.squeeze(0))

            return torch.stack(box_features_list, 0)

    def forward(self, obj_detect, label_img_list, roi_pool_output, representation_feature, previous_loc, curr_loc, 
                historical_reid=None, early_reid=None, curr_reid=None):
        if self.is_motion_model or self.is_motion_v2_model:
            pred_loc_wh, vis, vis_feature = self.motion_model(roi_pool_output, representation_feature, 
                                                              previous_loc, curr_loc, output_vis_feature=True)
        elif self.is_reid_model:
            assert historical_reid is not None and curr_reid is not None
            pred_loc_wh, vis, vis_feature = self.motion_model(historical_reid, curr_reid, roi_pool_output, 
                                                              representation_feature, previous_loc, curr_loc, output_vis_feature=True)
        elif self.is_simple_reid_model:
            assert early_reid is not None and curr_reid is not None
            pred_loc_wh, vis, vis_feature = self.motion_model(early_reid, curr_reid, roi_pool_output, 
                                                              representation_feature, previous_loc, curr_loc, output_vis_feature=True)
        elif self.is_simple_reid_v2_model:
            assert early_reid is not None and curr_reid is not None
            pred_loc_wh, vis, vis_feature = self.motion_model(early_reid, curr_reid, representation_feature, 
                                                              previous_loc, curr_loc, output_vis_feature=True)
        else:
            return

        pred_loc = wh_to_two_p(pred_loc_wh)
        roi_features = self.get_roi_features(obj_detect, label_img_list, pred_loc)

        roi_features = self.mlp_head(roi_features.flatten(start_dim=1))
        refinements = self.regress_head(roi_features) * self.regress_modulator(vis_feature)

        refined_pred_loc_wh = decode_motion(refinements, pred_loc_wh)

        return refined_pred_loc_wh, vis

