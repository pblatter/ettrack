# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import cv2
import json
import glob
import numpy as np
from os.path import join
from os import listdir

import argparse
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str, default='/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/trackingnet', help='your vid data dir')
args = parser.parse_args()

dataDir = args.dir

def main(dataDir='.', dataCropDir='./trackingnet_cropped'):
    trackingnet = []

    video_grandfathers = ['TRAIN_0', 'TRAIN_1', 'TRAIN_2', 'TRAIN_3']
    print(f'grandfathers: {video_grandfathers}')
    s = []
    for _, video_f in enumerate(video_grandfathers):

        try:
            video_fathers = sorted(listdir(join(dataDir, video_f, 'frames')))

            for vi, video in enumerate(video_fathers):

                try:
                    print('father class: {} video id: {:04d} / {:04d}'.format(video_f, vi, len(video_fathers)))
                    v = dict()
                    v['base_path'] = join(video_f, 'frames', video)
                    v['frame'] = []
                    video_base_path = join(dataDir, video_f, 'frames', video)

                    gts_path = join(dataDir, video_f, 'anno', f'{video}.txt')

                    gts = np.loadtxt(open(gts_path, "rb"), delimiter=',')

                    # get image size
                    im_path = join(video_base_path, '0.jpg')
                    im = cv2.imread(im_path)
                    size = im.shape  # height, width
                    frame_sz = [size[1], size[0]]  # width,height

                    # get all im name
                    jpgs = glob.glob(join(video_base_path, '*.jpg'))
                    jpgs = sorted(jpgs, key=lambda path: int(path.split('/')[-1][:-4]))

                    f = dict()
                    for idx, img_path in enumerate(jpgs):
                        f['frame_sz'] = frame_sz
                        f['img_path'] = img_path.split('/')[-1]

                        gt = gts[idx]
                        bbox = [int(g) for g in gt]   # (x,y,w,h)
                        f['bbox'] = bbox
                        v['frame'].append(f.copy())

                    s.append(v)

                except Exception as e:
                    print(f'Exception while processing video {video_base_path} with groundtruth path {gts_path}')
                    print(f'Exception: {e}')
        except Exception as e:
            print(f"problem with list dir: {join(dataDir, video_f, 'frames')}")
            print(f'Exception e: {e}')
    trackingnet.append(s)

    print('save json (raw trackingnet info), please wait 1 min~')
    json.dump(trackingnet, open(join(dataCropDir, 'trackingnet.json'), 'w'), indent=4, sort_keys=True)
    print('trackingnet.json has been saved in ./')

if __name__ == '__main__':
    since = time.time()
    main(sys.argv[1], sys.argv[2])
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))