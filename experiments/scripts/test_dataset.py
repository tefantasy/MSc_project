from tracktor.datasets.mot17_tracks import MOT17Tracks
from tracktor.datasets.mot17_tracks_wrapper import MOT17TracksWrapper
from tracktor.datasets.mot17_vis import MOT17Vis

from tracktor.frcnn_fpn import FRCNN_FPN

from torch.utils.data import DataLoader

import torch

# dataset = MOT17TracksWrapper('train', 0.8, 0.25, 1)

# dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
dataset = MOT17Vis('train', 0.8, vis_threshold=0.0)

dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

print(len(dataloader))

# for data in dataloader:
#     # print(data['gt'], data['vis'], data['start_frame'], data['last_frame'])
#     # print(label['gt'], label['vis'], label['frame'])
#     print(data['img'], data['gt'].size(), data['vis'].size())
#     break
##################################################


obj_detect = FRCNN_FPN(num_classes=2)
obj_detect.load_state_dict(torch.load('output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model',
                               map_location=lambda storage, loc: storage))
obj_detect.eval()
obj_detect.cuda()

for data in dataloader:
    obj_detect.load_image(data['img'])
    # print(obj_detect.preprocessed_images.size())
    print(obj_detect.__dict__.keys())


    break