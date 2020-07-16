import os
import os.path as osp

import numpy as np

from .mot_sequence import MOT17Sequence



class MOT17Tracks(MOT17Sequence):
    """
    Load MOT17 in the way that basic items
      of the dataset are the continuous tracks.
    """

    def __init__(self, seq_name, vis_threshold, track_len, keep_short_track=False, simple_reid=False):
        super().__init__(seq_name, vis_threshold=vis_threshold)

        self._num_frames = len(self.data)
        self._track_len = track_len
        self._keep_short_track = keep_short_track
        self._simple_reid = simple_reid

        track_data = self.build_tracks()
        self._track_data = self.build_samples(track_data)

    def build_tracks(self):
        finished_tracks = []
        unfinished_tracks = {}

        for i, frame_data in enumerate(self.data, start=1):
            for id in frame_data['gt'].keys():
                if id in unfinished_tracks:
                    # extend existing tracks
                    unfinished_tracks[id]['gt'].append(frame_data['gt'][id])
                    unfinished_tracks[id]['im_path'].append(frame_data['im_path'])
                    unfinished_tracks[id]['vis'].append(frame_data['vis'][id])

                    unfinished_tracks[id]['last_frame'] = i
                else:
                    # start new tracks
                    unfinished_tracks[id] = {}

                    unfinished_tracks[id]['gt'] = [ frame_data['gt'][id] ]
                    unfinished_tracks[id]['im_path'] = [ frame_data['im_path'] ]
                    unfinished_tracks[id]['vis'] = [ frame_data['vis'][id] ]

                    unfinished_tracks[id]['start_frame'] = i
                    unfinished_tracks[id]['last_frame'] = i

            # stop old tracks
            for id in list(unfinished_tracks.keys()):
                if unfinished_tracks[id]['last_frame'] < i:
                    if unfinished_tracks[id]['last_frame'] - unfinished_tracks[id]['start_frame'] + 1 >= self._track_len \
                            or self._keep_short_track:
                        finished_tracks.append(unfinished_tracks[id])

                    del unfinished_tracks[id]

        # stop all the remaining tracks
        for id in list(unfinished_tracks.keys()):
            if unfinished_tracks[id]['last_frame'] - unfinished_tracks[id]['start_frame'] + 1 >= self._track_len \
                    or self._keep_short_track:
                finished_tracks.append(unfinished_tracks[id])

            del unfinished_tracks[id]

        return finished_tracks

    def build_samples(self, track_data):
        segmented_tracks = []

        for track in track_data:
            if track['last_frame'] - track['start_frame'] + 1 <= self._track_len:
                if self._keep_short_track:
                    segmented_tracks.append(track)
            else:
                offset = 0
                while track['start_frame'] + offset + self._track_len - 1 <= track['last_frame']:
                    track_clip = {
                        'gt' : track['gt'][offset:offset+self._track_len],
                        'im_path' : track['im_path'][offset:offset+self._track_len],
                        'vis' : track['vis'][offset:offset+self._track_len],
                        'start_frame' : track['start_frame'] + offset,
                        'last_frame' : track['start_frame'] + offset + self._track_len - 1
                    }
                    if self._simple_reid:
                        early_len = min(5, offset + 1)
                        track_clip['early_start_frame'] = track['start_frame']
                        track_clip['early_im_path'] = track['im_path'][:early_len]
                        track_clip['early_gt'] = track['gt'][:early_len]

                    segmented_tracks.append(track_clip)
                    offset += 1

        return segmented_tracks

    def __len__(self):
        return len(self._track_data)

    def __getitem__(self, idx):
        return self._track_data[idx]