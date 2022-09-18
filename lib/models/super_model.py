import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class Super_model(nn.Module):
    def __init__(self, search_size=255, template_size=127, stride=16):
        super(Super_model, self).__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.stride = stride
        self.score_size = round(self.search_size / self.stride)
        self.num_kernel = round(self.template_size / self.stride) ** 2
        self.criterion = nn.BCEWithLogitsLoss()
        self.retrain = False

    def feature_extractor(self, x, cand_b=None):
        '''cand_b: candidate path for backbone'''
        # if isinstance(self, nn.DataParallel):
        #     return self.features.module.forward_backbone(x, cand_b, stride=self.stride)
        # else:
        #     return self.features.forward_backbone(x, cand_b, stride=self.stride)
        if self.retrain:
            return self.features(x, stride=self.stride)
        else:
            return self.features(x, cand_b, stride=self.stride)

    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.score_size
        print('grids size=',sz)

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2
        self.grid_to_search_y = y * self.stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0)#.cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0)#.cuda()

        #self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)
        #self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)

        self.grid_to_search_x = self.grid_to_search_x.repeat(32, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(32, 1, 1, 1)

    def template(self, z, cand_b):
        self.zf = self.feature_extractor(z, cand_b)

        if self.neck is not None:
            self.zf = self.neck(self.zf, crop=False)


    def track(self, x, cand_b, cand_h_dict):
        # supernet backbone
        xf = self.feature_extractor(x, cand_b)
        # dim adjust
        if self.neck is not None:
            xf = self.neck(xf)
        # feature adjustment and correlation
        feat_dict = self.feature_fusor(self.zf, xf)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict)
        return oup

    def forward(self, template, search, label=None, reg_target=None, reg_weight=None,
                cand_b=None, cand_h_dict=None):

        '''run siamese network'''
        zf = self.feature_extractor(template, cand_b=cand_b)
        xf = self.feature_extractor(search, cand_b=cand_b)

        if self.neck is not None:
            zf = self.neck(zf, crop=False)
            xf = self.neck(xf, crop=False)

        # feature adjustment and correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict=cand_h_dict)
        if label is not None and reg_target is not None and reg_weight is not None:
            # compute loss
            reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
            cls_loss = self._weighted_BCE(oup['cls'], label)
            return cls_loss, reg_loss
        else:
            return feat_dict

    def _soft_learnable_BCE(self, pred, label, soft_weights, mode='all'):
        pred = pred.view(-1) # [32,1,16,16] -> [8192]
        label = label.view(-1)
        
        dim = label.shape[0]
        ext_factor = dim // soft_weights.shape[0]
        soft_weights = soft_weights.repeat_interleave(ext_factor)

        if mode == 'pos' or mode == 'all':
            pos = label.data.eq(1).nonzero().squeeze()
            loss_pos = self._learnable_weighted_cls_loss(pred, label, pos, soft_weights)
        if mode == 'neg' or mode == 'all':
            neg = label.data.eq(0).nonzero().squeeze()
            loss_neg = self._learnable_weighted_cls_loss(pred, label, neg, soft_weights)
        # return
        if mode == 'pos':
            return loss_pos
        elif mode == 'neg':
            return loss_neg
        elif mode == 'all':
            return loss_pos * 0.5 + loss_neg * 0.5

    def _learnable_weighted_cls_loss(self, pred, label, select, soft_weights):
        '''
        BCEWithLogitsLoss but able to backpropagate through the weights
        '''
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        soft_weights = torch.index_select(soft_weights, 0, select)

        def _criterion(pred, target, soft_weights):
            
            
            #pred=F.sigmoid(pred)
            #tmp = -(target * torch.log(pred) + (1-target) * torch.log(1-pred))
            
            m = nn.LogSigmoid()
            tmp = -(target * m(pred) + (1-target) * m(-pred))
            tmp *= soft_weights
            loss=torch.mean(tmp)
            return loss
        #criterion = torch.nn.BCEWithLogitsLoss(weight=soft_weights)
        return _criterion(pred, label, soft_weights)  # the same as tf version


    def _soft_weighted_BCE(self, pred, label, soft_weights, mode='all'):
        pred = pred.view(-1) # [32,1,16,16] -> [8192]
        label = label.view(-1)
        
        dim = label.shape[0]
        ext_factor = dim // soft_weights.shape[0]
        soft_weights = soft_weights.repeat_interleave(ext_factor)

        if mode == 'pos' or mode == 'all':
            pos = label.data.eq(1).nonzero().squeeze()
            loss_pos = self._soft_cls_loss(pred, label, pos, soft_weights)
        if mode == 'neg' or mode == 'all':
            neg = label.data.eq(0).nonzero().squeeze()
            loss_neg = self._soft_cls_loss(pred, label, neg, soft_weights)
        # return
        if mode == 'pos':
            return loss_pos
        elif mode == 'neg':
            return loss_neg
        elif mode == 'all':
            return loss_pos * 0.5 + loss_neg * 0.5

    def _soft_cls_loss(self, pred, label, select, soft_weights):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        soft_weights = torch.index_select(soft_weights, 0, select)
        criterion = torch.nn.BCEWithLogitsLoss(weight=soft_weights)
        return criterion(pred, label)  # the same as tf version
    
    def _weighted_BCE(self, pred, label, mode='all'):
        pred = pred.view(-1)
        label = label.view(-1)
        if mode == 'pos' or mode == 'all':
            pos = label.data.eq(1).nonzero().squeeze()
            loss_pos = self._cls_loss(pred, label, pos)
        if mode == 'neg' or mode == 'all':
            neg = label.data.eq(0).nonzero().squeeze()
            loss_neg = self._cls_loss(pred, label, neg)
        # return
        if mode == 'pos':
            return loss_pos
        elif mode == 'neg':
            return loss_neg
        elif mode == 'all':
            return loss_pos * 0.5 + loss_neg * 0.5

    def _elementwise_weighted_BCE(self, pred, label, mode='all', device='cuda'):

        batch_size, s1, s2 = label.shape
        
        pred = pred.view(-1)
        label = label.view(-1)

        index_tensor = torch.ones(batch_size,s1,s2).to(device)
        for i in range(batch_size):
            index_tensor[i,:] *= i
        index_tensor = index_tensor.view(-1)

        return_losses = torch.zeros(batch_size).to(device)

        # go over all the elements
        for i in range(batch_size):

            # build a mask for the data 
            mask = index_tensor.eq(i)
            label_i = label[mask]
            pred_i = pred[mask]

            if mode == 'pos' or mode == 'all':
                # pos is the index of the elements equal to one inside the data
                pos = label_i.data.eq(1).nonzero().squeeze() 
                loss_pos = self._cls_loss(pred_i, label_i, pos)
            if mode == 'neg' or mode == 'all':
                neg = label_i.data.eq(0).nonzero().squeeze()
                loss_neg = self._cls_loss(pred_i, label_i, neg)
            # return
            if mode == 'pos':
                return_losses[i] = loss_pos
            elif mode == 'neg':
                return_losses[i] = loss_neg
            elif mode == 'all':
                return_losses[i] = loss_pos * 0.5 + loss_neg * 0.5

        return return_losses

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)  # the same as tf version

    def add_weighted_iouloss(self, bbox_pred, reg_target, reg_weight, soft_weights, iou_mode='iou'):
        """

        :param bbox_pred:
        :param reg_target:
        :param reg_weight:
        :param grid_x:  used to get real target bbox
        :param grid_y:  used to get real target bbox
        :return:
        """
        assert (iou_mode == 'iou' or iou_mode == 'diou')

        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)

        dim = reg_weight_flatten.shape[0]
        ext_factor = dim // soft_weights.shape[0]

        soft_weights = soft_weights.repeat_interleave(ext_factor)

        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]
        soft_weights = soft_weights[pos_inds]

        if iou_mode == 'iou':
            loss = self._IOULoss(bbox_pred_flatten, reg_target_flatten, soft_weights)
        elif iou_mode == 'diou':
            loss = self._DIoU_Loss(bbox_pred_flatten, reg_target_flatten)
        else:
            raise ValueError ('iou_mode should be iou or diou')
        return loss

    def add_iouloss_element_wise(self, bbox_pred, reg_target, reg_weight, iou_mode='iou', device='cuda'):
        """

        :param bbox_pred:
        :param reg_target:
        :param reg_weight:
        :param grid_x:  used to get real target bbox
        :param grid_y:  used to get real target bbox
        :return:
        """
        assert (iou_mode == 'iou' or iou_mode == 'diou')
        
        #print("bbox shape: ", bbox_pred.shape)
        #print("target shape: ", reg_target.shape)
        #print("reg weight shape: ", reg_weight.shape)

        b, s1, s2, c = reg_target.shape

        index_tensor = torch.ones(b,s1,s2).to(device)
        for i in range(b):
            index_tensor[i,:] *= i

        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        #print("flattened pred: ", bbox_pred_flatten.shape)
        reg_target_flatten = reg_target.reshape(-1, 4)
        #print("flattened target: ", reg_target_flatten.shape)
        reg_weight_flatten = reg_weight.reshape(-1)
        #print("flattened reg weight: ", reg_weight_flatten.shape)
        pos_inds = torch.nonzero(reg_weight_flatten > 0)
        #print("pre-squeeze pos_inds shape: ", pos_inds.shape)
        pos_inds = pos_inds.squeeze(1)
        #print("post-squeeze pos_inds shape: ", pos_inds.shape)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        #print("bbox pred flatten 2: ", bbox_pred_flatten.shape)
        reg_target_flatten = reg_target_flatten[pos_inds]
        #print("reg target flatten: ", reg_target_flatten.shape)
        index_flattened = index_tensor.reshape(-1)[pos_inds]


        if iou_mode == 'iou':
            loss = self._elementwise_IOULoss(bbox_pred_flatten, reg_target_flatten, index_flattened, batch_size=b, device=device)
        elif iou_mode == 'diou':
            loss = self._DIoU_Loss(bbox_pred_flatten, reg_target_flatten)
        else:
            raise ValueError ('iou_mode should be iou or diou')
        return loss


    def _elementwise_IOULoss(self, pred, target, index_tensor, batch_size, weight=None, device='cuda'):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        # pred and target both \in [pos_inds]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0)) # \in [pos_inds]

        #print(f'n elements in elementwise losses: {losses.shape}')
        #print(f'elementwise loss mean: {torch.mean(losses)}')
        #print(f'losses: {losses}')

        return_losses = torch.zeros(batch_size).to(device) # \in [32] for batch_size 32
        #print(f'return losses: {return_losses}')
        #print(f'index tensor: {index_tensor}')
        for i in range(batch_size):
            
            mask = index_tensor.eq(i)
            l = losses[mask]
            if l.numel() == 0:
                return_losses[i] = 0.0

            else: 
                return_losses[i] = torch.mean(l)
        
        return return_losses

        
    
    def add_iouloss(self, bbox_pred, reg_target, reg_weight, iou_mode='iou'):
        """

        :param bbox_pred:
        :param reg_target:
        :param reg_weight:
        :param grid_x:  used to get real target bbox
        :param grid_y:  used to get real target bbox
        :return:
        """
        assert (iou_mode == 'iou' or iou_mode == 'diou')
        
        #print("bbox shape: ", bbox_pred.shape)
        #print("target shape: ", reg_target.shape)
        #print("reg weight shape: ", reg_weight.shape)

        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        #print("flattened: ", bbox_pred_flatten.shape)
        reg_target_flatten = reg_target.reshape(-1, 4)
        #print("flattened: ", reg_target_flatten.shape)
        reg_weight_flatten = reg_weight.reshape(-1)
        #print("flattened: ", reg_weight_flatten.shape)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)
        #print("pos_inds: ", pos_inds)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        #print("bbox pred flatten 2: ", bbox_pred_flatten.shape)
        reg_target_flatten = reg_target_flatten[pos_inds]
        #print("reg target flatten: ", reg_target_flatten.shape)
        if iou_mode == 'iou':
            loss = self._IOULoss(bbox_pred_flatten, reg_target_flatten)
        elif iou_mode == 'diou':
            loss = self._DIoU_Loss(bbox_pred_flatten, reg_target_flatten)
        else:
            raise ValueError ('iou_mode should be iou or diou')
        return loss

    def _IOULoss(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]      # distance to the left border of the bbox
        target_top = target[:, 1]       # distance to the top border of the bbox
        target_right = target[:, 2]     # distance to the right border of the bbox
        target_bottom = target[:, 3]    # distance to the bottom border of the bbox

        # pred and target both \in [pos_inds]

        target_area = (target_left + target_right) * (target_top + target_bottom) # still \in [pos_inds]
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)           # still \in [pos_inds]

        # for every single of the pos_inds elements -> compute the minimum distance to the left and right border
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right) 
        # for every single of the pos_inds elements -> compute the minimum distance to the bottom and top border
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect # \in [pos_inds]
        area_union = target_area + pred_area - area_intersect 

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0)) # \in [pos_inds]
        

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

    def add_mining_iouloss(self, bbox_pred, reg_target, reg_weight, iou_mode='iou'):
        """

        :param bbox_pred:
        :param reg_target:
        :param reg_weight:
        :param grid_x:  used to get real target bbox
        :param grid_y:  used to get real target bbox
        :return:
        """
        assert (iou_mode == 'iou' or iou_mode == 'diou')
        
        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]
        if iou_mode == 'iou':
            loss = self._MiningIOULoss(bbox_pred_flatten, reg_target_flatten)
        elif iou_mode == 'diou':
            loss = self._DIoU_Loss(bbox_pred_flatten, reg_target_flatten)
        else:
            raise ValueError ('iou_mode should be iou or diou')
        return loss

    def _MiningIOULoss(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]      # distance to the left border of the bbox
        target_top = target[:, 1]       # distance to the top border of the bbox
        target_right = target[:, 2]     # distance to the right border of the bbox
        target_bottom = target[:, 3]    # distance to the bottom border of the bbox

        # pred and target both \in [pos_inds]

        target_area = (target_left + target_right) * (target_top + target_bottom) # still \in [pos_inds]
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)           # still \in [pos_inds]

        # for every single of the pos_inds elements -> compute the minimum distance to the left and right border
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right) 
        # for every single of the pos_inds elements -> compute the minimum distance to the bottom and top border
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect # \in [pos_inds]
        area_union = target_area + pred_area - area_intersect 

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0)) # \in [pos_inds]
        

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            #assert losses.numel() != 0
            if losses.numel() != 0:
                return losses.mean()
            else: 
                return "no posinds"

    def _IOULoss_debug(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]      # distance to the left border of the bbox
        target_top = target[:, 1]       # distance to the top border of the bbox
        target_right = target[:, 2]     # distance to the right border of the bbox
        target_bottom = target[:, 3]    # distance to the bottom border of the bbox

        # pred and target both \in [pos_inds]

        target_area = (target_left + target_right) * (target_top + target_bottom) # still \in [pos_inds]
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)           # still \in [pos_inds]

        # for every single of the pos_inds elements -> compute the minimum distance to the left and right border
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right) 
        # for every single of the pos_inds elements -> compute the minimum distance to the bottom and top border
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect # \in [pos_inds]
        area_union = target_area + pred_area - area_intersect 

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0)) # \in [pos_inds]
        print(f'IOU losses: {losses.shape}')
        print(f'IOU loss mean: {torch.mean(losses)}')
        print(f'losses: {losses}')

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return (losses.mean(), losses)

    def pred_to_image(self, bbox_pred):
        self.grid_to_search_x = self.grid_to_search_x.to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)  # 17*17
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)  # 17*17
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)  # 17*17
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)  # 17*17

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred

'''2020.09.11 for compute MACs'''
class Super_model_MACs(nn.Module):
    def __init__(self, search_size=255, template_size=127, stride=16):
        super(Super_model_MACs, self).__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.stride = stride
        self.score_size = round(self.search_size / self.stride)
        self.num_kernel = round(self.template_size / self.stride) ** 2

    def feature_extractor(self, x, cand_b):
        '''cand_b: candidate path for backbone'''
        if isinstance(self, nn.DataParallel):
            return self.features.module.forward_backbone(x, cand_b, stride=self.stride)
        else:
            return self.features.forward_backbone(x, cand_b, stride=self.stride)


    def forward(self, zf, search, cand_b, cand_h_dict):

        '''run siamese network'''
        xf = self.feature_extractor(search, cand_b)

        if self.neck is not None:
            xf = self.neck(xf, crop=False)

        # feature adjustment and correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict)

        return oup


'''2020.10.18 for retrain the searched model'''
class Super_model_retrain(Super_model):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(Super_model_retrain, self).__init__(search_size=search_size, template_size=template_size, stride=stride)

    def template(self, z):
        self.zf = self.features(z)

    def track(self, x):
        # supernet backbone
        xf = self.features(x)
        if self.neck is not None:
            raise ValueError ('neck should be None')
        # Point-wise Correlation
        feat_dict = self.feature_fusor(self.zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        return oup

    def forward(self, template, search, label, reg_target, reg_weight):
        '''backbone_index: which layer's feature to use'''
        zf = self.features(template, stride=self.stride)
        xf = self.features(search, stride=self.stride)
        if self.neck is not None:
            raise ValueError ('neck should be None')
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        # compute loss
        reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
        cls_loss = self._weighted_BCE(oup['cls'], label)
        return cls_loss, reg_loss
