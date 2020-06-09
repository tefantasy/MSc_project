import torch

def encode_motion(previous_loc, curr_loc):
    """
    *_loc: (batch, 4)
            Using (x_center, y_center, w, h) coordinates. 
    """

    motion_x = (curr_loc[:, [0]] - previous_loc[:, [0]]) / previous_loc[:, [2]]
    motion_y = (curr_loc[:, [1]] - previous_loc[:, [1]]) / previous_loc[:, [3]]
    motion_w = torch.log(curr_loc[:, [2]] / previous_loc[:, [2]])
    motion_h = torch.log(curr_loc[:, [3]] / previous_loc[:, [3]])

    return torch.cat([motion_x, motion_y, motion_w, motion_h], 1)

def decode_motion(motion, curr_loc):
    pred_loc_x = motion[:, [0]] * curr_loc[:, [2]] + curr_loc[:, [0]]
    pred_loc_y = motion[:, [1]] * curr_loc[:, [3]] + curr_loc[:, [1]]
    pred_loc_w = torch.exp(motion[:, [2]]) * curr_loc[:, [2]]
    pred_loc_h = torch.exp(motion[:, [3]]) * curr_loc[:, [3]]

    return torch.cat([pred_loc_x, pred_loc_y, pred_loc_w, pred_loc_h], 1)

def two_p_to_wh(loc):
    loc_w = loc[:, [2]] - loc[:, [0]] + 1.0
    loc_h = loc[:, [3]] - loc[:, [1]] + 1.0
    loc_center_x = loc[:, [0]] + 0.5 * loc_w
    loc_center_y = loc[:, [1]] + 0.5 * loc_h

    return torch.cat([loc_center_x, loc_center_y, loc_w, loc_h], 1)

def wh_to_two_p(loc):
    loc_x1 = loc[:, [0]] - 0.5 * loc[:, [2]]
    loc_y1 = loc[:, [1]] - 0.5 * loc[:, [3]]
    loc_x2 = loc[:, [0]] + 0.5 * loc[:, [2]] - 1.0
    loc_y2 = loc[:, [1]] - 0.5 * loc[:, [3]] - 1.0

    return torch.cat([loc_x1, loc_y1, loc_x2, loc_y2], 1)