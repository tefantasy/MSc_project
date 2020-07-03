import os
import os.path as osp
import math

import numpy as np

from .mot_sequence import MOT17Sequence

class MOT17Clips(MOT17Sequence):
    """
    Load MOT17 in the way that basic items of the dataset are 
    clips of k-consecutive frames and tracks in the corresponding clips.
    """
    def __init__(self, seq_name, split, train_ratio, vis_threshold, clip_len, min_track_len=2):
        """
        min_track_len: the minimum length of a track (except the last one, label).
        """
        super().__init__(seq_name, vis_threshold=0.0)

        self.num_frames = len(self.data)
        self.clip_len = clip_len
        self.min_track_len = min_track_len

        val_start_frame = int(math.floor(self.num_frames * train_ratio))
        self.seq_clip_data = []

        if split == 'train':
            range_start, range_end = clip_len, val_start_frame + 1
        elif split == 'val':
            range_start, range_end = val_start_frame + clip_len, self.num_frames + 1

        for end_frame in range(range_start, range_end):
            clip_tracks = self.build_clip_tracks(end_frame - clip_len, clip_len)
            im_paths = [self.data[frame]['im_path'] for frame in range(end_frame - clip_len, end_frame)]
            
            self.seq_clip_data.append({
                'seq': self._seq_name,
                'start_frame': end_frame - clip_len + 1, # frame id starts at 1
                'im_paths': im_paths,
                'tracks': clip_tracks
            })



    def build_clip_tracks(self, begin_frame, clip_len):
        finished_tracks = []
        unfinished_tracks = {}

        for i, frame_data in enumerate(self.data[begin_frame:begin_frame+clip_len]):
            for id in frame_data['gt'].keys():
                if id in unfinished_tracks:
                    # extend existing tracks
                    unfinished_tracks[id]['gt'].append(frame_data['gt'][id])
                    unfinished_tracks[id]['frame_offset'].append(i)
                    unfinished_tracks[id]['vis'].append(frame_data['vis'][id])
                else:
                    # start new tracks
                    unfinished_tracks[id] = {}

                    unfinished_tracks[id]['gt'] = [ frame_data['gt'][id] ]
                    unfinished_tracks[id]['frame_offset'] = [ i ]
                    unfinished_tracks[id]['vis'] = [ frame_data['vis'][id] ]

            # stop old tracks
            for id in list(unfinished_tracks.keys()):
                if unfinished_tracks[id]['frame_offset'][-1] < i:
                    if unfinished_tracks[id]['frame_offset'][-1] - unfinished_tracks[id]['frame_offset'][0] + 1 > self.min_track_len:
                        finished_tracks.append(unfinished_tracks[id])

                    del unfinished_tracks[id]

        # stop all the remaining tracks
        for id in list(unfinished_tracks.keys()):
            if unfinished_tracks[id]['frame_offset'][-1] - unfinished_tracks[id]['frame_offset'][0] + 1 > self.min_track_len:
                finished_tracks.append(unfinished_tracks[id])

            del unfinished_tracks[id]

        return finished_tracks

    def __len__(self):
        return len(self.seq_clip_data)

    def __getitem__(self, idx):
        return self.seq_clip_data[idx]