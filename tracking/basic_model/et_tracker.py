import torch

from lib.models.super_model_DP import Super_model_DP
from lib.models.model_parts import *
import lib.models.models as lighttrack_model
from lib.utils.utils import load_lighttrack_model
from tracking.basic_model.exemplar_transformer import ExemplarTransformer


class ET_Tracker(Super_model_DP):

    def __init__(self, linear_reg=True, 
                    search_size=256, 
                    template_size=128, 
                    stride=16, 
                    adj_channel=128, 
                    e_exemplars=4,
                    path_name='back_04502514044521042540+cls_211000022+reg_100000111_ops_32',
                    arch='LightTrackM_Subnet',
                    sm_normalization=False,
                    temperature=1,
                    dropout=False):
        super(ET_Tracker, self).__init__()

        '''
        Args:
            - sm_normalization: whether to normalize the QK^T by sqrt(C) in the MultiheadTransConver
        '''

        self.backbone_path_name = path_name

        # Backbone network
        siam_net = lighttrack_model.__dict__[arch](path_name, stride=stride)

        # Backbone
        self.backbone_net = siam_net.features

        # Neck
        self.neck = MC_BN(inp_c=[96])  # BN with multiple types of input channels

        # Feature Fusor
        self.feature_fusor = Point_Neck_Mobile_simple_DP(num_kernel_list=[64], matrix=True,
                                                             adj_channel=adj_channel)  # stride=8, stride=16

        inchannels = 128
        outchannels_cls = 256
        outchannels_reg = 192

        padding_3 = (3 - 1) // 2
        padding_5 = (5 - 1) // 2

        self.cls_branch_1 = SeparableConv2d_BNReLU(inchannels, outchannels_cls, kernel_size=5, stride=1, padding=padding_5)
        self.cls_branch_2 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls, dw_padding=padding_5, e_exemplars=e_exemplars, dw_kernel_size=5, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.cls_branch_3 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.cls_branch_4 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.cls_branch_5 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.cls_branch_6 = SeparableConv2d_BNReLU(outchannels_cls, outchannels_cls, kernel_size=3, stride=1, padding=padding_3)
        self.cls_pred_head = cls_pred_head(inchannels=outchannels_cls)

        

        self.bbreg_branch_1 = SeparableConv2d_BNReLU(inchannels, outchannels_reg, kernel_size=3, stride=1, padding=padding_3)
        self.bbreg_branch_2 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.bbreg_branch_3 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.bbreg_branch_4 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.bbreg_branch_5 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.bbreg_branch_6 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_5, e_exemplars=e_exemplars, dw_kernel_size=5, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.bbreg_branch_7 = ExemplarTransformer(in_channels=outchannels_reg, out_channels=outchannels_reg, dw_padding=padding_5, e_exemplars=e_exemplars, dw_kernel_size=5, pw_kernel_size=1, sm_normalization=sm_normalization, temperature=temperature, dropout=dropout)
        self.bbreg_branch_8 = SeparableConv2d_BNReLU(outchannels_reg, outchannels_reg, kernel_size=5, stride=1, padding=padding_5)
        self.reg_pred_head = reg_pred_head(inchannels=outchannels_reg, linear_reg=linear_reg)



    def forward(self, template, search, label=None, reg_target=None, reg_weight=None):
        
        torch.autograd.set_detect_anomaly(True)

        # extract features 
        zf = self.backbone_net(template)
        xf = self.backbone_net(search)  

        # Batch Normalization before Corr
        zf, xf = self.neck(zf, xf) 

        # pixelwise correlation
        feat_dict = self.feature_fusor(zf, xf) 

        c = self.cls_branch_1(feat_dict['cls'])
        c = self.cls_branch_2(c)
        c = self.cls_branch_3(c)
        c = self.cls_branch_4(c)
        c = self.cls_branch_5(c)
        c = self.cls_branch_6(c)
        c = self.cls_pred_head(c)
        
        b = self.bbreg_branch_1(feat_dict['reg'])
        b = self.bbreg_branch_2(b) 
        b = self.bbreg_branch_3(b) 
        b = self.bbreg_branch_4(b) 
        b = self.bbreg_branch_5(b) 
        b = self.bbreg_branch_6(b) 
        b = self.bbreg_branch_7(b) 
        b = self.bbreg_branch_8(b)
        b = self.reg_pred_head(b)

        oup = {}
        oup['cls'] = c
        oup['reg'] = b
       
        if label is not None and reg_target is not None and reg_weight is not None:
            
            # compute loss
            reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight) # scalar
            cls_loss = self._weighted_BCE(oup['cls'], label) # scalar
            return cls_loss, reg_loss


        return oup

    def initialize(self, model_name, checkpoint_epoch=None):
        '''
        Initializes the network, i.e. loads the model from a checkpoint.
        '''
        load_lighttrack_model(model=self, model_name=model_name, checkpoint_epoch=checkpoint_epoch)
        print(f'model initializing successful')

    def template(self, z):
        '''
        Used during the tracking -> computes the embedding of the target in the first frame.
        '''
        self.zf = self.backbone_net(z)

    def track(self, x):
       
        xf = self.backbone_net(x)  

        # Batch Normalization before Corr
        zf, xf = self.neck(self.zf, xf) 

        # pixelwise correlation
        feat_dict = self.feature_fusor(zf, xf) 

        c = self.cls_branch_1(feat_dict['cls'])
        c = self.cls_branch_2(c) 
        c = self.cls_branch_3(c) 
        c = self.cls_branch_4(c) 
        c = self.cls_branch_5(c) 
        c = self.cls_branch_6(c)
        c = self.cls_pred_head(c)
        
        b = self.bbreg_branch_1(feat_dict['reg'])
        b = self.bbreg_branch_2(b) 
        b = self.bbreg_branch_3(b) 
        b = self.bbreg_branch_4(b) 
        b = self.bbreg_branch_5(b) 
        b = self.bbreg_branch_6(b) 
        b = self.bbreg_branch_7(b) 
        b = self.bbreg_branch_8(b)
        b = self.reg_pred_head(b)
       


        oup = {}
        oup['cls'] = c
        oup['reg'] = b

        return oup['cls'], oup['reg']

    
