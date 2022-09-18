import os
import cv2
import yaml
import numpy as np

import torch
import torch.nn.functional as F
from lib.utils.utils import load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid, python2round
from lib.utils.utils import load_pretrain, cxy_wh_2_rect
'''2020.09.19 inherit base class'''
from toolkit.got10k.trackers import Tracker
'''2020.09.21 SuperTracker for GOT-10K'''
class SuperTracker(Tracker):
    def __init__(self, info, name, effi=0, model=None, cand=None):
        super(SuperTracker, self).__init__(name=name)
        self.info = info   # model and benchmark info
        self.align = info.align
        self.online = info.online
        self.trt = info.TRT
        self.effi = effi
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224,0.225]).view(3, 1, 1)
        self.model = model
        assert isinstance(cand,tuple) and isinstance(cand[0],list) and isinstance(cand[1],dict)
        self.DP = (len(cand) == 3)
        if self.DP:
            self.cand_b, self.cand_h, self.cand_op = cand
            if self.cand_op[0] == 1:
                self.stride = 8
            elif self.cand_op[0] == 2 or self.cand_op[0] == 3:
                self.stride = 16
        else:
            self.cand_b, self.cand_h = cand
            self.stride = info.stride

    def normalize(self,x):
        ''' input is in (C,H,W) format'''
        x /= 255
        x -= self.mean
        x /= self.std
        return x

    def init(self, im, box, hp=None):
        '''initialize'''
        '''parse iamge'''
        im = np.array(im)
        x1,y1,w,h = box
        target_pos = np.array([x1 + w / 2, y1 + h / 2])
        target_sz = np.array([w, h])

        model = self.model
        # in: whether input infrared image
        self.state = dict()
        # epoch test
        p = OceanConfig(stride=self.stride, effi=self.effi)

        self.state['im_h'] = im.shape[0]
        self.state['im_w'] = im.shape[1]

        self.grids(p)   # self.grid_to_search_x, self.grid_to_search_y

        net = model

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        z_crop = self.normalize(z_crop)
        z = z_crop.unsqueeze(0)
        '''specify path'''
        if self.DP:
            net.template(z.cuda(), self.cand_b, self.cand_op)
        else:
            net.template(z.cuda(), self.cand_b)

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))

        self.state['p'] = p
        self.state['net'] = net
        self.state['avg_chans'] = avg_chans
        self.state['window'] = window
        self.state['target_pos'] = target_pos
        self.state['target_sz'] = target_sz

        return self.state

    def forward(self, net, x_crops, target_pos, target_sz, window, scale_z, p, debug=False):
        if self.DP:
            oup = net.track(x_crops, self.cand_b, self.cand_h, self.cand_op)
        else:
            oup = net.track(x_crops, self.cand_b, self.cand_h)
        cls_score, bbox_pred = oup['cls'], oup['reg']
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

    def update(self, im, online_score=None, gt=None):
        '''Tracking Function'''

        im = np.array(im) # PIL to numpy

        p = self.state['p']
        net = self.state['net']
        avg_chans = self.state['avg_chans']
        window = self.state['window']
        target_pos = self.state['target_pos']
        target_sz = self.state['target_sz']

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans)
        self.state['x_crop'] = x_crop.clone() # torch float tensor, (3,H,W)
        x_crop = self.normalize(x_crop).unsqueeze(0)

        target_pos, target_sz, _ = self.forward(net, x_crop.cuda(), target_pos, target_sz * scale_z,
                                                              window, scale_z, p)
        target_pos[0] = max(0, min(self.state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(self.state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(self.state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(self.state['im_h'], target_sz[1]))
        self.state['target_pos'] = target_pos
        self.state['target_sz'] = target_sz
        self.state['p'] = p

        location = cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])
        return location

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
