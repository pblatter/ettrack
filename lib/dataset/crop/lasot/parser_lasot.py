# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import cv2
import json
import glob
import numpy as np
from os.path import join
from os import listdir
import time
import sys

def main(dataDir='.', dataCropDir='./got10k_cropped'):
    lasot = []

    videos_fathers = sorted(listdir(dataDir))
    s = []
    for _, video_f in enumerate(videos_fathers):

        try:
            videos_sons = sorted(listdir(join(dataDir, video_f)))

            for vi, video in enumerate(videos_sons):

                try:
                    print('father class: {} video id: {:04d} / {:04d}'.format(video_f, vi, len(videos_sons)))
                    v = dict()
                    v['base_path'] = join(video_f, video)
                    v['frame'] = []
                    video_base_path = join(dataDir, video_f, video)
                    gts_path = join(video_base_path, 'groundtruth.txt')
                    gts = np.loadtxt(open(gts_path, "rb"), delimiter=',')

                    # get image size
                    im_path = join(video_base_path, 'img', '00000001.jpg')
                    im = cv2.imread(im_path)
                    size = im.shape  # height, width
                    frame_sz = [size[1], size[0]]  # width,height

                    # get all im name
                    jpgs = sorted(glob.glob(join(video_base_path, 'img', '*.jpg')))

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
            print(f'problem with list dir: {join(dataDir, video_f)}')
            print(f'Exception e: {e}')
    lasot.append(s)

    print('save json (raw lasot info), please wait 1 min~')
    json.dump(lasot, open(join(dataCropDir, 'lasot.json'), 'w'), indent=4, sort_keys=True)
    print('lasot.json has been saved')

if __name__ == '__main__':
    since = time.time()
    main(sys.argv[1], sys.argv[2])
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
