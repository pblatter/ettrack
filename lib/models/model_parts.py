import math
import torch
import torch.nn as nn
from collections import OrderedDict

class SeparableConv2d_BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d_BNReLU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.ReLU(self.BN(x))
        return x

'''Channel attention module'''
class CAModule(nn.Module):

    def __init__(self, channels=64, reduction=1):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


def pixel_corr(Kernel_tmp, Feature, KERs=None):
    size = Kernel_tmp.size()
    CORR = []
    for i in range(len(Feature)):
        ker = Kernel_tmp[i:i + 1]
        fea = Feature[i:i + 1]
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
        ker = ker.unsqueeze(2).unsqueeze(3)
        if not (type(KERs) == type(None)):
            ker = torch.cat([ker, KERs[i]], 0)
        co = F.conv2d(fea, ker.contiguous())
        CORR.append(co)
    corr = torch.cat(CORR, 0)
    return corr

def pixel_corr_mat(z, x):
    '''
    z is kernel ([32, 96, 8, 8])
    x is search ([32, 96, 16, 16])

    z -> (32, 64, 96)
    x -> (32, 96, 256)
    '''
    b, c, h, w = x.size()
    z_mat = z.view((b,c,-1)).transpose(1,2) # (b,64,c)
    x_mat = x.view((b,c,-1)) # (b,c,256)
    # matmul = (b,64,256) == every pixel of z is correlated with every pixel of x
    return torch.matmul(z_mat, x_mat).view((b,-1,h,w)) # (32, 64, 16, 16)

class GroupPW(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self, num_channel, cat=False, CA=True, matrix=False):
        super(GroupPW, self).__init__()
        # self.weight = nn.Parameter(torch.ones(3))
        self.cat = cat
        self.CA = CA
        self.matrix = matrix
        if self.CA:
            self.CA_layer = CAModule(channels=num_channel)


    def forward(self, z, x):
        z11 = z[0] # kernel (template)
        # print(z11.size())
        x11 = x[0] # search
        # print(x11.size())
        '''pixel-wise correlation'''
        '''2020.09.16 Matrix Mul Version'''
        if self.matrix:
            re11 = pixel_corr_mat(z11, x11)
        else:
            re11 = pixel_corr(z11, x11)
        if self.CA:
            '''channel attention'''
            s = self.CA_layer(re11)
            if self.cat:
                return torch.cat([s,x11],dim=1)
            else:
                return s
        else:
            return re11

# neck
class MC_BN(nn.Module):
    def __init__(self, inp_c=(40,80,96,256)):
        super(MC_BN, self).__init__()
        self.BN_z = nn.ModuleList() # BN for the template branch
        self.BN_x = nn.ModuleList() # BN for the search branch
        for idx, channel in enumerate(inp_c):
            self.BN_z.append(nn.BatchNorm2d(channel))
            self.BN_x.append(nn.BatchNorm2d(channel))
    def forward(self, kernel, search, index=None):
        if index is None:
            index = -1
        return self.BN_z[index](kernel), self.BN_x[index](search)


# feature fusor
class Point_Neck_Mobile_simple_DP(nn.Module):
    def __init__(self, num_kernel_list=(256,64), cat=False, matrix=True, adjust=True, adj_channel=128):
        super(Point_Neck_Mobile_simple_DP, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = torch.nn.ModuleList()
        self.adj_layer = torch.nn.ModuleList()
        for num_kernel in num_kernel_list:
            self.pw_corr.append(GroupPW(num_kernel, cat=cat, CA=True, matrix=matrix))
            self.adj_layer.append(nn.Conv2d(num_kernel, adj_channel, 1))

    def forward(self, kernel, search, stride_idx=None):
        '''stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16'''
        if stride_idx is None:
            stride_idx = -1
        oup = {}
        corr_feat = self.pw_corr[stride_idx]([kernel], [search])
        #print("corr_feat shape: ", corr_feat.shape)
        #print(f'type of corr_feat: {type(corr_feat)}')
        if self.adjust:
            corr_feat = self.adj_layer[stride_idx](corr_feat)
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
        
class tower_supernet_singlechannel(nn.Module):
    """
    tower's supernet
    """
    def __init__(self, inchannels=256, outchannels=256, towernum=8,
                 base_op=SeparableConv2d_BNReLU, kernel_list=[3,5,0]):
        super(tower_supernet_singlechannel, self).__init__()
        if 0 in kernel_list:
            assert (kernel_list[-1] == 0)
        self.kernel_list = kernel_list
        self.num_choice = len(self.kernel_list)

        self.tower = nn.ModuleList()

        # tower
        for i in range(towernum):
            '''the first layer, we don't use identity'''
            if i == 0:
                op_list = nn.ModuleList()
                if self.num_choice == 1:
                    kernel_size = self.kernel_list[-1]
                    padding = (kernel_size - 1) // 2
                    op_list.append(base_op(inchannels, outchannels, kernel_size=kernel_size,
                                           stride=1, padding=padding))
                else:
                    for choice_idx in range(self.num_choice-1):
                        kernel_size = self.kernel_list[choice_idx]
                        padding = (kernel_size - 1) // 2
                        op_list.append(base_op(inchannels, outchannels, kernel_size=kernel_size,
                                               stride=1, padding=padding))
                self.tower.append(op_list)

            else:
                op_list = nn.ModuleList()
                for choice_idx in range(self.num_choice):
                    kernel_size = self.kernel_list[choice_idx]
                    if kernel_size != 0:
                        padding = (kernel_size - 1) // 2
                        op_list.append(base_op(outchannels, outchannels, kernel_size=kernel_size,
                                               stride=1, padding=padding))
                    else:
                        op_list.append(Identity())
                self.tower.append(op_list)

    def forward(self, x, arch_list):

        for archs, arch_id in zip(self.tower, arch_list):
            x = archs[arch_id](x)

        return x

# head supernet
class head_supernet(nn.Module):
    def __init__(self, channel_list=[112,256,512], kernel_list=[3,5,0], inchannels=64, towernum=8, linear_reg=False,
                 base_op_name='SeparableConv2d_BNReLU'):
        super(head_supernet, self).__init__()
        if base_op_name == 'SeparableConv2d_BNReLU':
            base_op = SeparableConv2d_BNReLU
        elif base_op_name == 'Conv2d_BNReLU':
            base_op = Conv2d_BNReLU
        else:
            raise ValueError ('Unsupported OP')
        self.num_cand = len(channel_list)
        self.cand_tower_cls = nn.ModuleList()
        self.cand_head_cls = nn.ModuleList()
        self.cand_tower_reg = nn.ModuleList()
        self.cand_head_reg = nn.ModuleList()
        self.tower_num = towernum
        # cls
        for outchannel in channel_list:
            self.cand_tower_cls.append(tower_supernet_singlechannel(inchannels=inchannels, outchannels=outchannel,
                                                                    towernum=towernum, base_op=base_op,
                                                                    kernel_list=kernel_list))
            self.cand_head_cls.append(cls_pred_head(inchannels=outchannel))
        # reg
        for outchannel in channel_list:
            self.cand_tower_reg.append(tower_supernet_singlechannel(inchannels=inchannels, outchannels=outchannel,
                                                                    towernum=towernum, base_op=base_op,
                                                                    kernel_list=kernel_list))
            self.cand_head_reg.append(reg_pred_head(inchannels=outchannel, linear_reg=linear_reg))
    def forward(self, inp, cand_dict=None):
        '''cand_dict key: cls, reg
         [0/1/2, []]'''
        if cand_dict is None:
            cand_dict = {'cls':[0,[0]*self.tower_num], 'reg':[0,[0]*self.tower_num]}
        oup = {}
        # cls
        cand_list_cls = cand_dict['cls'] # [0/1/2, []]
        cls_feat = self.cand_tower_cls[cand_list_cls[0]](inp['cls'],cand_list_cls[1])
        oup['cls'] = self.cand_head_cls[cand_list_cls[0]](cls_feat)
        # reg
        cand_list_reg = cand_dict['reg'] # [0/1/2, []]
        reg_feat = self.cand_tower_cls[cand_list_reg[0]](inp['reg'],cand_list_reg[1])
        oup['reg'] = self.cand_head_reg[cand_list_reg[0]](reg_feat)

        return oup

class cls_pred_head(nn.Module):
    def __init__(self, inchannels=256):
        super(cls_pred_head, self).__init__()
        self.cls_pred = nn.Conv2d(inchannels, 1, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        '''mode should be in ['all', 'cls', 'reg']'''
        x = 0.1 * self.cls_pred(x)
        return x

class reg_pred_head(nn.Module):
    def __init__(self, inchannels=256, linear_reg=False, stride=16):
        super(reg_pred_head, self).__init__()
        self.linear_reg = linear_reg
        self.stride = stride
        # reg head
        self.bbox_pred = nn.Conv2d(inchannels, 4, kernel_size=3, stride=1, padding=1)
        # adjust scale
        if not self.linear_reg:
            self.adjust = nn.Parameter(0.1 * torch.ones(1))
            self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())
    def forward(self, x):
        if self.linear_reg:
            x = nn.functional.relu(self.bbox_pred(x)) * self.stride
        else:
            x = self.adjust * self.bbox_pred(x) + self.bias
            x = torch.exp(x)
        return x

class Conv2d_BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(Conv2d_BNReLU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
    def forward(self, x):
        return self.ReLU(self.BN(self.conv1(x)))


