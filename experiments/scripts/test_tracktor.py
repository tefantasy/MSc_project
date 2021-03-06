import os
import time
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.tracker_neural_mm import TrackerNeuralMM
from tracktor.motion.model import MotionModel
from tracktor.motion.model_v2 import MotionModelV2
from tracktor.motion.backbone_model import BackboneMotionModel
from tracktor.motion.model_reid import MotionModelReID
from tracktor.motion.model_simple_reid import MotionModelSimpleReID
from tracktor.motion.model_simple_reid_v2 import MotionModelSimpleReIDV2
from tracktor.motion.model_v3 import MotionModelV3
from tracktor.motion.vis_simple_reid import VisSimpleReID
from tracktor.motion.refine_model import RefineModel
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


@ex.automain
def main(tracktor, reid, _config, _log, _run):
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector.")

    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    # neural motion model 

    vis_model = VisSimpleReID()

    motion_model = MotionModelV3(vis_model)
    motion_model.load_state_dict(torch.load('output/motion/finetune_motion_model_v3.pth')) 

    motion_model.eval()
    motion_model.cuda()

    save_vis_results = False

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        # tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])
        tracker = TrackerNeuralMM(obj_detect, reid_network, motion_model, tracktor['tracker'], save_vis_results=save_vis_results, vis_model=None)

    time_total = 0
    num_frames = 0
    mot_accums = []
    dataset = Datasets(tracktor['dataset'], {'use_val_split':True})
    for seq in dataset:
        tracker.reset()

        start = time.time()

        _log.info(f"Tracking: {seq}")

        data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        for i, frame in enumerate(tqdm(data_loader)):
            if len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
                with torch.no_grad():
                    tracker.step(frame)
                num_frames += 1
        results = tracker.get_results()

        time_total += time.time() - start

        _log.info(f"Tracks found: {len(results)}")
        _log.info(f"Runtime for {seq}: {time.time() - start :.1f} s.")

        if tracktor['interpolate']:
            results = interpolate(results)

        if seq.no_gt:
            _log.info(f"No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        _log.info(f"Writing predictions to: {output_dir}")
        seq.write_results(results, output_dir)
        if save_vis_results:
            vis_results = tracker.get_vis_results()
            seq.write_vis_results(vis_results, output_dir)

        if tracktor['write_images']:
            plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(seq)))

    _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
              f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")
    if mot_accums:
        evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt], generate_overall=True)
