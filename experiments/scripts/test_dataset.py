from tracktor.datasets.mot17_tracks import MOT17Tracks
from tracktor.datasets.mot17_tracks_wrapper import MOT17TracksWrapper
from tracktor.datasets.mot17_vis import MOT17Vis

from tracktor.frcnn_fpn import FRCNN_FPN

from torchvision.models.detection.transform import resize_boxes

from torch.utils.data import DataLoader

import torch

dataset = MOT17TracksWrapper('train', 0.8, 0.25, input_track_len=8, max_sample_frame=2)

dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
# dataset = MOT17Vis('train', 0.8, vis_threshold=0.0)

# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print(len(dataloader))

# print(dataset._track_data[0])
# print(dataset._track_label[0])

for data, label in dataloader:
    # print(data['gt'], data['vis'], data['start_frame'], data['last_frame'])
    # print(label['gt'], label['vis'], label['frame'])
    print(data['img'].size(), data['gt'].size(), data['vis'].size())
    print(label['img'].size(), label['gt'].size(), label['vis'].size())
    break
##################################################


# obj_detect = FRCNN_FPN(num_classes=2)
# obj_detect.load_state_dict(torch.load('output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model',
#                                map_location=lambda storage, loc: storage))
# obj_detect.eval()
# obj_detect.cuda()

# for data in dataloader:
#     obj_detect.load_image(data['img'])
#     # print(obj_detect.preprocessed_images[0].size())  # ImageList
#     # print(obj_detect.features)                       # OrderedDict of multiple scale features

#     gts = data['gt'].squeeze(0).cuda()
#     gts = resize_boxes(gts, obj_detect.original_image_sizes[0], obj_detect.preprocessed_images.image_sizes[0])
#     gts = [gts]

#     box_features = obj_detect.roi_heads.box_roi_pool(obj_detect.features, gts, obj_detect.preprocessed_images.image_sizes)
#     print(box_features.size())                         # (num_bbox, dim=256, 7, 7)
#     box_head_features = obj_detect.roi_heads.box_head(box_features)
#     print(box_head_features.size())                    # (num_bbox, dim=1024)
#     print(data['gt'].size())

#     break