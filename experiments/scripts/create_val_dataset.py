import csv
import os
import os.path as osp
import shutil
import configparser
import math

from tracktor.config import cfg

def process_gt_file(source, target, train_len):
    f_in = open(source, 'r')
    f_out = open(target, 'w')

    for line in f_in:
        splitted_line = line.split(',')
        frame_id = int(splitted_line[0])
        if frame_id <= train_len:
            continue
        splitted_line[0] = str(frame_id - train_len)
        f_out.write(','.join(splitted_line))

    f_in.close()
    f_out.close()


source_data_dir = osp.join(cfg.DATA_DIR, 'MOT17Det', 'train')
source_label_dir = osp.join(cfg.DATA_DIR, 'MOT17Labels', 'train')

base_dir = osp.join(cfg.DATA_DIR, 'MOT17-val')
data_dir = osp.join(base_dir, 'MOT17Det')
label_dir = osp.join(base_dir, 'MOT17Labels')


if osp.exists(base_dir):
    shutil.rmtree(base_dir)

os.mkdir(base_dir)
os.mkdir(data_dir)
os.mkdir(label_dir)
os.mkdir(osp.join(data_dir, 'test'))
os.mkdir(osp.join(label_dir, 'test'))

data_dir = osp.join(data_dir, 'train')
label_dir = osp.join(label_dir, 'train')
os.mkdir(data_dir)
os.mkdir(label_dir)

train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
                         'MOT17-11', 'MOT17-13']
det = 'FRCNN'

for seq_name in train_folders:
    # process data dir
    source_seq_dir = osp.join(source_data_dir, seq_name)
    out_seq_dir = osp.join(data_dir, seq_name)
    os.mkdir(out_seq_dir)

    config = configparser.ConfigParser()
    config.read(osp.join(source_seq_dir, 'seqinfo.ini'))
    seq_len = int(config['Sequence']['seqLength'])

    train_len = int(math.floor(seq_len * 0.8))
    val_len = seq_len - train_len
    config['Sequence']['seqLength'] = str(val_len)
    with open(osp.join(out_seq_dir, 'seqinfo.ini'), 'w') as f:
        config.write(f, space_around_delimiters=False)

    out_img_dir = osp.join(out_seq_dir, 'img1')
    os.mkdir(out_img_dir)

    for i in range(val_len):
        frame_id = train_len + i + 1
        shutil.copy(osp.join(source_seq_dir, 'img1', "{:06d}.jpg".format(frame_id)), 
                    osp.join(out_img_dir, "{:06d}.jpg".format(i+1)))

    os.mkdir(osp.join(out_seq_dir, 'gt'))
    process_gt_file(osp.join(source_seq_dir, 'gt', 'gt.txt'), osp.join(out_seq_dir, 'gt', 'gt.txt'), train_len)

    # process label dir
    source_seq_dir = osp.join(source_label_dir, seq_name+'-'+det)
    out_seq_dir = osp.join(label_dir, seq_name+'-'+det)
    os.mkdir(out_seq_dir)

    os.mkdir(osp.join(out_seq_dir, 'det'))
    process_gt_file(osp.join(source_seq_dir, 'det', 'det.txt'), osp.join(out_seq_dir, 'det', 'det.txt'), train_len)

    print('Finished %s' % seq_name)