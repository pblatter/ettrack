from os.path import join, exists
from os import makedirs
import argparse

import lib.dataset.crop.coco.par_crop_coco as par_crop_coco
import lib.dataset.crop.coco.gen_json_coco as gen_json_coco

import lib.dataset.crop.got10k.parser_got10k as parser_got10k
import lib.dataset.crop.got10k.par_crop_got10k as par_crop_got10k
import lib.dataset.crop.got10k.gen_json_got10k as gen_json_got10k

import lib.dataset.crop.lasot.parser_lasot as parser_lasot
import lib.dataset.crop.lasot.par_crop_lasot as par_crop_lasot
import lib.dataset.crop.lasot.gen_json_lasot as gen_json_lasot

import lib.dataset.crop.trackingnet.parse_trackingnet as parse_trackingnet
import lib.dataset.crop.trackingnet.par_crop_trackingnet as par_crop_trackingnet
import lib.dataset.crop.trackingnet.gen_json_trackingnet as gen_json_trackingnet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir',type=str, default='./datasets', help='your dataset dir')
parser.add_argument('--dataset_crop_dir',type=str, default='./datasets_cropped', help='the dir for the new dataset')
parser.add_argument('--instanc_size',type=int, default=511, help='instance size')
parser.add_argument('--num_threads',type=int, default=16, help='number of threads')
args = parser.parse_args()

datasetsPath = args.dataset_dir
datasetsCroppedPath = args.dataset_crop_dir
instanc_size = args.instanc_size
num_threads = args.num_threads

def makeDir(path):
    if not exists(path): makedirs(path)
    return path

# COCO
cocoPath = makeDir(join(datasetsPath, 'coco'))
cocoCroppedPath = makeDir(join(datasetsCroppedPath, 'coco'))

par_crop_coco.main(cocoPath, dataCropDir=cocoCroppedPath, instanc_size=instanc_size, num_threads=num_threads)
gen_json_coco.main(cocoPath, cocoCroppedPath)

# GOT10k
got10kPath = makeDir(join(datasetsPath, 'got10k'))
got10kCroppedPath = makeDir(join(datasetsCroppedPath, 'got10k'))

parser_got10k.main(got10kPath, got10kCroppedPath)
par_crop_got10k.main(got10kPath, got10kCroppedPath, instanc_size=instanc_size, num_threads=num_threads)
gen_json_got10k.main(got10kCroppedPath)

# LASOT
lasotPath = makeDir(join(datasetsPath, 'LaSOTBenchmark'))
lasotCroppedPath = makeDir(join(datasetsCroppedPath, 'LaSOTBenchmark'))

parser_lasot.main(lasotPath, lasotCroppedPath)
par_crop_lasot.main(lasotPath, lasotCroppedPath, instanc_size=instanc_size, num_threads=num_threads)
gen_json_lasot.main(lasotCroppedPath)

# TrackingNet
trackingnetPath = makeDir(join(datasetsPath, 'trackingnet'))
trackingnetCroppedPath = makeDir(join(datasetsCroppedPath, 'trackingnet'))

parse_trackingnet.main(trackingnetPath, trackingnetCroppedPath)
par_crop_trackingnet.main(trackingnetPath, trackingnetCroppedPath, instanc_size=instanc_size, num_threads=num_threads)
gen_json_trackingnet.main(trackingnetCroppedPath)