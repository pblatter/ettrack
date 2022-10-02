from pycocotools.coco import COCO
from os.path import join
import json
import os
import time
import sys

def main(dataDir='.', dataCropDir='./coco_cropped'):
    for dataType in ['val2017', 'train2017']:
        dataset = dict()
        annFile = join(dataDir, 'annotations/instances_{}.json'.format(dataType))
        coco = COCO(annFile)
        n_imgs = len(coco.imgs)
        for n, img_id in enumerate(coco.imgs):
            print('subset: {} image id: {:04d} / {:04d}'.format(dataType, n, n_imgs))
            img = coco.loadImgs(img_id)[0]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)
            video_crop_base_path = join(dataType, img['file_name'].split('/')[-1].split('.')[0])

            if len(anns) > 0:
                dataset[video_crop_base_path] = dict()

            for trackid, ann in enumerate(anns):
                rect = ann['bbox']
                c = ann['category_id']
                bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
                if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                    continue
                dataset[video_crop_base_path]['{:02d}'.format(trackid)] = {'000000': bbox}

        print('save json (dataset), please wait a few seconds~')
        json.dump(dataset, open('{}.json'.format(os.path.join(dataCropDir, dataType)), 'w'), indent=4, sort_keys=True)
        print('done!')

if __name__ == '__main__':
    since = time.time()
    main(sys.argv[1])
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
