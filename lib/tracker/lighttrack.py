import os
import cv2
import yaml
import numpy as np

import torch
import torch.nn.functional as F
from lib.utils.utils import load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid, python2round

'''2020.08.10 Normalize input image'''
class Lighttrack(object):
    def __init__(self, info, effi=0):
        super(Lighttrack, self).__init__()
        self.info = info   # model and benchmark info
        self.stride = info.stride
        self.effi = effi
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224,0.225]).view(3, 1, 1)

    def normalize(self,x):
        ''' input is in (C,H,W) format'''
        x /= 255
        x -= self.mean
        x /= self.std
        return x
    def init(self, im, target_pos, target_sz, model, hp=None, cfg_outer=None):
        # in: whether input infrared image
        state = dict()
        # epoch test
        p = OceanConfig(stride=self.stride, effi=self.effi)

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        # single test
        if not hp and not self.info.epoch_test:
            prefix = [x for x in ['OTB', 'VOT'] if x in self.info.dataset]
            if len(prefix) == 0: prefix = [self.info.dataset]
            absPath = os.path.abspath(os.path.dirname(__file__))
            yname = 'LightTrack.yaml'
            # yname = 'Ocean.yaml'
            yamlPath = os.path.join(absPath, '../../experiments/test/{0}/'.format(prefix[0]), yname)
            cfg = load_yaml(yamlPath)
            cfg_benchmark = cfg[self.info.dataset]
            # print('cfg file loaded. cfg is shown below.\n',cfg)
            '''if not epoch test, we replace the original cfg with new one'''
            p.update(cfg_benchmark)
            p.renew()

            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = cfg_benchmark['big_sz']
                p.renew()
            else:
                p.instance_size = cfg_benchmark['small_sz']
                p.renew()
        '''Specially process LASOT (2020.11.12 Have a try)'''
        if self.info.dataset == 'LASOT':
            prefix = [x for x in ['OTB', 'VOT', 'LASOT'] if x in self.info.dataset]
            if len(prefix) == 0: prefix = [self.info.dataset]
            absPath = os.path.abspath(os.path.dirname(__file__))
            yname = 'LightTrack.yaml'
            # yname = 'Ocean.yaml'
            yamlPath = os.path.join(absPath, '../../experiments/test/{0}/'.format(prefix[0]), yname)
            cfg = load_yaml(yamlPath)
            cfg_benchmark = cfg[self.info.dataset]
            '''if not epoch test, we replace the original cfg with new one'''
            p.update(cfg_benchmark)
            p.renew()

            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = cfg_benchmark['big_sz']
                p.renew()
            else:
                p.instance_size = cfg_benchmark['small_sz']
                p.renew()
        if cfg_outer is not None:
            p.update(cfg_outer)
            p.renew()

            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = cfg_outer['big_sz']
                p.renew()
            else:
                p.instance_size = cfg_outer['small_sz']
                p.renew()
            print('''Instance Size: %d'''%(p.instance_size))
        # double check
        # print('======= hyper-parameters: penalty_k: {}, wi: {}, lr: {}, ratio: {}, instance_sz: {}, score_sz: {} ======='.format(p.penalty_k, p.window_influence, p.lr, p.ratio, p.instance_size, p.score_size))

        # param tune
        if hp:
            p.update(hp)
            p.renew()

            # for small object (from DaSiamRPN released)
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = hp['big_sz']
                p.renew()
            else:
                p.instance_size = hp['small_sz']
                p.renew()

        self.grids(p)   # self.grid_to_search_x, self.grid_to_search_y

        net = model # the siamese network including heads (my model)

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans) # get a subwindow of the same size as the template
        z_crop = self.normalize(z_crop) # normalize cropped input image
        z = z_crop.unsqueeze(0)
        net.template(z.cuda()) # this simply calls the feature extractor -> what for?

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        return state

    def update(self, net, x_crops, target_pos, target_sz, window, scale_z, p, debug=False):

        cls_score, bbox_pred = net.track(x_crops)
        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2-pred_x1, pred_y2-pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2-pred_x1) / (pred_y2-pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])
        if debug:
            return target_pos, target_sz, cls_score[r_max, c_max], cls_score
        else:
            return target_pos, target_sz, cls_score[r_max, c_max]

    def track(self, state, im, online_score=None, gt=None):
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
      
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

      

        x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans)
        
        state['x_crop'] = x_crop.clone() # torch float tensor, (3,H,W)
        x_crop = self.normalize(x_crop)
        x_crop = x_crop.unsqueeze(0)
        debug = True
        if debug:
            target_pos, target_sz, _, cls_score = self.update(net, x_crop.cuda(), target_pos, target_sz*scale_z, window, scale_z, p, debug=debug)
            state['cls_score'] = cls_score
        else:
            target_pos, target_sz, _ = self.update(net, x_crop.cuda(), target_pos, target_sz * scale_z,
                                                              window, scale_z, p, debug=debug)
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p

        return state

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        # print('ATTENTION',p.instance_size,p.score_size)
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2


    def IOUgroup(self, pred_x1, pred_y1, pred_x2, pred_y2, gt_xyxy):
        # overlap

        x1, y1, x2, y2 = gt_xyxy

        xx1 = np.maximum(pred_x1, x1)  # 17*17
        yy1 = np.maximum(pred_y1, y1)
        xx2 = np.minimum(pred_x2, x2)
        yy2 = np.minimum(pred_y2, y2)

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        area = (x2 - x1) * (y2 - y1)

        target_a = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        inter = ww * hh
        overlap = inter / (area + target_a - inter)

        return overlap

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


class OceanConfig(object):
    def __init__(self, stride=8, effi=0):
        self.penalty_k = 0.062
        self.window_influence = 0.38
        self.lr = 0.765
        self.windowing = 'cosine'
        if effi:
            self.exemplar_size = 128
            self.instance_size = 256
        else:
            self.exemplar_size = 127
            self.instance_size = 255
        # total_stride = 8
        # score_size = (instance_size - exemplar_size) // total_stride + 1 + 8  # for ++
        self.total_stride = stride
        self.score_size = int(round(self.instance_size / self.total_stride))
        self.context_amount = 0.5
        self.ratio = 0.94


    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        # self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8 # for ++
        self.score_size = int(round(self.instance_size / self.total_stride))
