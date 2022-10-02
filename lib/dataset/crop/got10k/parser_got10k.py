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
    sub_sets = sorted({'train'})

    got10k = []
    for sub_set in sub_sets:
        sub_set_base_path = join(dataDir, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        s = []
        for vi, video in enumerate(videos):
            try:
                print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))
                v = dict()
                v['base_path'] = join(sub_set, video)
                v['frame'] = []
                video_base_path = join(sub_set_base_path, video)
                gts_path = join(video_base_path, 'groundtruth.txt')
                gts = np.loadtxt(open(gts_path, "rb"), delimiter=',')

                # get image size
                im_path = join(video_base_path, '00000001.jpg')
                im = cv2.imread(im_path)
                size = im.shape  # height, width
                frame_sz = [size[1], size[0]]  # width,height

                # get all im name
                jpgs = sorted(glob.glob(join(video_base_path, '*.jpg')))

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
        got10k.append(s)
    print('save json (raw got10k info), please wait for about 1 min')
    json.dump(got10k, open(join(dataCropDir, 'got10k.json'), 'w'), indent=4, sort_keys=True)
    print('got10k.json has been saved')

if __name__ == '__main__':
    since = time.time()
    main(sys.argv[1], sys.argv[2])
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
