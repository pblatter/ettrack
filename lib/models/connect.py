import torch
import torch.nn as nn
import torch.nn.functional as F

'''2020.09.11 hswish'''
from .mobilenetv3 import h_swish


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
    b, c, h, w = x.size()
    z_mat = z.view((b,c,-1)).transpose(1,2) # (b,64,c)
    x_mat = x.view((b,c,-1)) # (b,c,256)
    return torch.matmul(z_mat, x_mat).view((b,-1,h,w)) # (b,64,256)


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

class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, pixelshuffle=False):
        super(AdjustLayer, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
        self.pixelshuffle = pixelshuffle
        if self.pixelshuffle:
            self.PS_module = nn.Sequential(
                nn.Conv2d(in_channels, in_channels*2, kernel_size=1),
                nn.PixelShuffle(2)
            )
            self.downsample = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, crop=False):
        if self.upsample:
            x = self.upsample_layer(x)
            x = nn.functional.interpolate(x,scale_factor=2,mode='bilinear')
        '''pixel shuffle'''
        if self.pixelshuffle:
            x = self.PS_module(x)
        '''num channel adjustment'''
        x_ori = self.downsample(x)
        if x_ori.size(3) < 20 and crop:
            l = 4
            r = -4
            xf = x_ori[:, :, l:r, l:r]

        if not crop:
            return x_ori
        else:
            return x_ori, xf


class matrix_light(nn.Module):
    """
    encode backbone feature
    """
    '''depth-wise separable'''
    def __init__(self, in_channels, out_channels, activation='relu', BN_choice=None):
        super(matrix_light, self).__init__()
        if activation == 'relu':
            non_linear = nn.ReLU(inplace=True)
        elif activation == 'hswish':
            non_linear = h_swish()
        else:
            raise ValueError('Unsupported activation type.')
        # same size (11)
        if BN_choice == 'perm':
            self.matrix11_k = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                non_linear,
                nn.BatchNorm2d(out_channels)
            )
            self.matrix11_s = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                non_linear,
                nn.BatchNorm2d(out_channels)
            )
        elif BN_choice == 'before':
            self.matrix11_k = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                non_linear,
                nn.BatchNorm2d(out_channels)
            )
            self.matrix11_s = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                non_linear,
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.matrix11_k = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                non_linear,
            )
            self.matrix11_s = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                non_linear,
            )



    def forward(self, z, x):
        z11 = self.matrix11_k(z)
        x11 = self.matrix11_s(x)
        return [z11], [x11]


'''2020.6.27 Pixel-wise fusion & channel attention'''
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
        z11 = z[0]
        # print(z11.size())
        x11 = x[0]
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


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
'''2020.10.31 Both depthwise and Pointwise use non-linearity'''
class SeparableConv2d_MV1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d_MV1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.relu(self.BN(self.conv1(x)))
        x = self.pointwise(x)
        return x
'''2020.09.10 Depthwise-Separable with residual'''
class SeparableConv2d_Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d_Residual, self).__init__()

        self.module = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        self.dim_conf = (in_channels != out_channels)
        if self.dim_conf:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    def forward(self, inp):
        if self.dim_conf:
            return self.module(inp) + self.residual(inp)
        else:
            return self.module(inp) + inp
'''2020.09.13 Point-Depth-Point'''
class PDP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(PDP, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, 1, 1, bias=bias),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding,
                      dilation, groups=mid_channels, bias=bias),
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        )

    def forward(self, x):
        return self.module(x)

'''2020.08.18 box_tower with mobile settings (depthwise separable)'''
class box_tower_light(nn.Module):
    """
    box tower for FCOS reg
    """
    def __init__(self, inchannels=512, outchannels=256, towernum=1,
                 num_kernel=None, kernel_sz = 3, dilation=1, cat=False,
                 block_name = 'DS', activation='relu',CA=True, matrix=False,
                 BN_choice=None, bbox_compute='exp', stride=16):
        super(box_tower_light, self).__init__()
        self.BN_choice = BN_choice
        self.num_kernel = num_kernel
        self.bbox_compute = bbox_compute
        self.stride = stride
        '''2020.09.10 support diverse building blocks'''
        if block_name == 'DS':
            block = SeparableConv2d
        elif block_name == 'DSR':
            block = SeparableConv2d_Residual
        elif block_name == 'PDP':
            block = PDP
        else:
            raise ValueError('Block should be DS or DSR')
        if dilation != 1:
            kernel_sz_eq = kernel_sz + (kernel_sz - 1) * (dilation - 1)
            padding = int((kernel_sz_eq - 1) / 2)
        else:
            padding = int((kernel_sz - 1) / 2)
        tower = []
        cls_tower = []
        # encode backbone
        self.cls_encode = matrix_light(in_channels=inchannels, out_channels=outchannels,
                                       activation=activation, BN_choice=BN_choice)
        self.reg_encode = matrix_light(in_channels=inchannels, out_channels=outchannels,
                                       activation=activation, BN_choice=BN_choice)
        self.cls_dw = GroupPW(num_kernel, cat=cat, CA=CA, matrix=matrix)
        self.reg_dw = GroupPW(num_kernel, cat=cat, CA=CA, matrix=matrix)

        if BN_choice == 'after':
            self.BN_cls = nn.BatchNorm2d(num_kernel)
            self.BN_reg = nn.BatchNorm2d(num_kernel)
        '''2020.08.25 specify kernel_sz and dilation for towers'''
        if activation == 'relu':
            non_linear = nn.ReLU()
        elif activation == 'hswish':
            non_linear = h_swish()
        else:
            raise ValueError('Unsupported activation function')
        # box pred head
        for i in range(towernum):
            if i == 0:
                if cat:
                    tower.append(block(num_kernel+outchannels, outchannels, kernel_size=kernel_sz,
                                        stride=1, padding=padding, dilation=dilation))  # 8x8=64 channels
                else:
                    # tower.append(nn.Conv2d(num_kernel, outchannels, kernel_size=3, stride=1, padding=1)) # 8x8=64 channels
                    tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz,
                                                 stride=1, padding=padding, dilation=dilation))  # 8x8=64 channels

            else:
                # tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
                tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1, padding=padding, dilation=dilation))

            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(non_linear)

        # cls tower
        for i in range(towernum):
            if i == 0:
                if cat:
                    cls_tower.append(block(num_kernel+outchannels, outchannels,
                                                     kernel_size=kernel_sz, stride=1,
                                                     padding=padding, dilation=dilation))
                else:
                    # cls_tower.append(nn.Conv2d(num_kernel, outchannels, kernel_size=3, stride=1, padding=1))
                    cls_tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz, stride=1, padding=padding, dilation=dilation))
            else:
                # cls_tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
                cls_tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1, padding=padding, dilation=dilation))

            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(non_linear)

        self.add_module('bbox_tower', nn.Sequential(*tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))


        # reg head
        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

        if bbox_compute == 'exp':
            # adjust scale
            self.adjust = nn.Parameter(0.1 * torch.ones(1))
            self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())
        elif bbox_compute == 'linear':
            pass

    def forward(self, search, kernel, update=None, mode='all'):
        '''mode should be in ['all', 'cls', 'reg']'''
        if mode not in ['cls','reg','all']:
            raise ValueError ("mode must be in ['cls','reg','all']")
        if mode == 'cls' or mode == 'all':
            # encode first
            if update is None:
                cls_z, cls_x = self.cls_encode(kernel, search)   # [z11, z12, z13]
            else:
                cls_z, cls_x = self.cls_encode(update, search)  # [z11, z12, z13]
            # Point-wise and channel attention
            cls_dw = self.cls_dw(cls_z, cls_x)
            if self.BN_choice == 'after':
                cls_dw = self.BN_cls(cls_dw)
            # cls tower
            c = self.cls_tower(cls_dw)
            cls = 0.1 * self.cls_pred(c)
        if mode == 'reg' or mode == 'all':
            # encode first
            reg_z, reg_x = self.reg_encode(kernel, search)  # [x11, x12, x13]
            # Point-wise and channel attention
            reg_dw = self.reg_dw(reg_z, reg_x)
            if self.BN_choice == 'after':
                reg_dw = self.BN_reg(reg_dw)
            # head
            x_reg = self.bbox_tower(reg_dw)
            if self.bbox_compute == 'exp':
                x = self.adjust * self.bbox_pred(x_reg) + self.bias
                x = torch.exp(x)
            elif self.bbox_compute == 'linear':
                x = self.stride * nn.functional.relu(self.bbox_pred(x_reg))
        # output
        if mode == 'cls':
            return cls
        elif mode == 'reg':
            return x
        elif mode == 'all':
            return x, cls, cls_dw, x_reg

'''2020.09.16 A simple head (no bottleneck, no adjust)'''
class box_tower_light_Simple(nn.Module):
    """
    box tower for FCOS reg
    """
    def __init__(self, outchannels=256, towernum=1,
                 num_kernel=None, kernel_sz = 3, dilation=1, cat=False,
                 CA=True, matrix=False, BN_choice=None, inchannels=112,
                 bbox_compute='exp', stride=16, block_name='DS'):
        super(box_tower_light_Simple, self).__init__()
        if block_name == 'DS':
            block = SeparableConv2d
        elif block_name == 'DS_MV1':
            block =SeparableConv2d_MV1
        else:
            raise ValueError ('Unsupported block name')
        non_linear = nn.ReLU()
        self.bbox_compute = bbox_compute
        self.stride = stride

        if dilation != 1:
            kernel_sz_eq = kernel_sz + (kernel_sz - 1) * (dilation - 1)
            padding = int((kernel_sz_eq - 1) / 2)
        else:
            padding = int((kernel_sz - 1) / 2)
        tower = []
        cls_tower = []
        '''BN before Pointwise Corr'''
        self.BN_choice = BN_choice
        if self.BN_choice == 'before':
            '''template and search use separate BN'''
            self.BN_adj_z = nn.BatchNorm2d(inchannels)
            self.BN_adj_x = nn.BatchNorm2d(inchannels)
        '''Point-wise Correlation'''
        self.pw_corr = GroupPW(num_kernel, cat=cat, CA=CA, matrix=matrix)

        # box pred head
        for i in range(towernum):
            if i == 0:
                tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz,
                                       stride=1, padding=padding, dilation=dilation))  # 8x8=64 channels

            else:
                tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1,
                                   padding=padding, dilation=dilation))

            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(non_linear)

        # cls tower
        for i in range(towernum):
            if i == 0:
                cls_tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz, stride=1, padding=padding, dilation=dilation))
            else:
                cls_tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1, padding=padding, dilation=dilation))

            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(non_linear)

        self.add_module('bbox_tower', nn.Sequential(*tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))


        # reg head
        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

        if bbox_compute == 'exp':
            # adjust scale
            self.adjust = nn.Parameter(0.1 * torch.ones(1))
            self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())
        elif bbox_compute == 'linear':
            pass
        else:
            raise ValueError ('Unsupported bbox compute method')

    def forward(self, search, kernel, update=None, mode='all'):
        '''mode should be in ['all', 'cls', 'reg']'''
        if mode not in ['cls','reg','all']:
            raise ValueError ("mode must be in ['cls','reg','all']")
        if self.BN_choice == 'before':
            kernel, search = self.BN_adj_z(kernel), self.BN_adj_x(search)
        corr_feat = self.pw_corr([kernel], [search])
        if mode == 'cls' or mode == 'all':
            # cls tower + Pred
            c = self.cls_tower(corr_feat)
            cls = 0.1 * self.cls_pred(c)
        if mode == 'reg' or mode == 'all':
            # Reg tower + Pred
            x_reg = self.bbox_tower(corr_feat)
            if self.bbox_compute == 'exp':
                x = self.adjust * self.bbox_pred(x_reg) + self.bias
                x = torch.exp(x)
            elif self.bbox_compute == 'linear':
                x = self.stride * nn.functional.relu(self.bbox_pred(x_reg))
            else:
                raise ValueError('Unsupported bbox compute method')
        # output
        if mode == 'cls':
            return cls
        elif mode == 'reg':
            return x
        elif mode == 'all':
            return x, cls, None, None

'''2020.10.2 FCOS head'''
class PW_FCOS_head(nn.Module):
    """
    Point-wise Corr + FCOS Head
    """
    def __init__(self, outchannels=256, towernum=1, num_kernel=None, kernel_sz = 3, dilation=1,
                 cat=False, CA=True, matrix=True, BN_choice='before', inchannels=112, stride=16):
        super(PW_FCOS_head, self).__init__()

        block = SeparableConv2d
        non_linear = nn.ReLU()
        self.stride = stride

        if dilation != 1:
            kernel_sz_eq = kernel_sz + (kernel_sz - 1) * (dilation - 1)
            padding = int((kernel_sz_eq - 1) / 2)
        else:
            padding = int((kernel_sz - 1) / 2)
        box_tower = [] # for reg and centerness
        cls_tower = [] # for cls
        '''BN before Pointwise Corr'''
        self.BN_choice = BN_choice
        if self.BN_choice == 'before':
            '''template and search use separate BN'''
            self.BN_adj_z = nn.BatchNorm2d(inchannels)
            self.BN_adj_x = nn.BatchNorm2d(inchannels)
        '''Point-wise Correlation'''
        self.pw_corr = GroupPW(num_kernel, cat=cat, CA=CA, matrix=matrix)

        # box pred head
        for i in range(towernum):
            if i == 0:
                box_tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz, stride=1,
                                       padding=padding, dilation=dilation))  # 8x8=64 channels

            else:
                box_tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1,
                                       padding=padding, dilation=dilation))

            box_tower.append(nn.BatchNorm2d(outchannels))
            box_tower.append(non_linear)

        # cls tower
        for i in range(towernum):
            if i == 0:
                cls_tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz, stride=1,
                                       padding=padding, dilation=dilation))
            else:
                cls_tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1,
                                       padding=padding, dilation=dilation))

            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(non_linear)

        self.add_module('box_tower', nn.Sequential(*box_tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))

        # pred head
        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cent_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(outchannels, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, search, kernel):
        if self.BN_choice == 'before':
            kernel, search = self.BN_adj_z(kernel), self.BN_adj_x(search)
        # Point-wise Corr
        corr_feat = self.pw_corr([kernel], [search])
        # tower
        x_cls, x_reg = self.cls_tower(corr_feat), self.box_tower(corr_feat)
        # pred
        cls, reg, cent = self.cls_pred(x_cls), self.bbox_pred(x_reg), self.cent_pred(x_reg)
        # post-processing
        reg = self.stride * nn.functional.relu(reg)

        return {'cls': cls, 'reg': reg, 'cent': cent}

'''2020.09.18 Let centerness and regression share the same conv tower'''
class box_tower_light_Simple_one_tower(nn.Module):
    """
    box tower for FCOS reg
    """

    def __init__(self, outchannels=256, towernum=1,
                 num_kernel=None, kernel_sz=3, dilation=1, cat=False,
                 CA=True, matrix=False, BN_choice=None, inchannels=112):
        super(box_tower_light_Simple_one_tower, self).__init__()

        block = SeparableConv2d
        non_linear = nn.ReLU()

        if dilation != 1:
            kernel_sz_eq = kernel_sz + (kernel_sz - 1) * (dilation - 1)
            padding = int((kernel_sz_eq - 1) / 2)
        else:
            padding = int((kernel_sz - 1) / 2)

        tower = nn.ModuleList()
        '''BN before Pointwise Corr'''
        self.BN_choice = BN_choice
        if self.BN_choice == 'before':
            self.BN_adj = nn.BatchNorm2d(inchannels)
        '''Point-wise Correlation'''
        self.pw_corr = GroupPW(num_kernel, cat=cat, CA=CA, matrix=matrix)

        # box pred head
        for i in range(towernum):
            if i == 0:
                tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz,
                                   stride=1, padding=padding, dilation=dilation))  # 8x8=64 channels

            else:
                tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1,
                                   padding=padding, dilation=dilation))

            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(non_linear)
        self.tower = nn.Sequential(*tower)

        # pred head
        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

    def forward(self, search, kernel, update=None, mode='all'):
        '''mode should be in ['all', 'cls', 'reg']'''
        if mode not in ['cls', 'reg', 'all']:
            raise ValueError("mode must be in ['cls','reg','all']")
        if self.BN_choice == 'before':
            kernel, search = self.BN_adj(kernel), self.BN_adj(search)
        corr_feat = self.pw_corr([kernel], [search])
        tower_feat = self.tower(corr_feat)
        if mode == 'cls' or mode == 'all':
            cls = 0.1 * self.cls_pred(tower_feat)
        if mode == 'reg' or mode == 'all':
            x = self.adjust * self.bbox_pred(tower_feat) + self.bias
            x = torch.exp(x)
        # output
        if mode == 'cls':
            return cls
        elif mode == 'reg':
            return x
        elif mode == 'all':
            return x, cls, None, None

'''2020.09.18 A simplified version (more easy to understand)'''
class simple_head(nn.Module):

    def __init__(self, outchannels=256, towernum=8,
                 num_kernel=None, CA=True, matrix=True):
        super(simple_head, self).__init__()

        block = SeparableConv2d
        kernel_sz, padding, dilation = 3, 1, 1

        tower = []
        cls_tower = []

        '''Point-wise Correlation'''
        self.pw_corr = GroupPW(num_kernel, CA=CA, matrix=matrix)

        non_linear = nn.ReLU()

        # box tower
        for i in range(towernum):
            if i == 0:
                tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz,
                                   stride=1, padding=padding, dilation=dilation))  # 8x8=64 channels
            else:
                tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1, padding=padding,
                                   dilation=dilation))
            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(non_linear)

        # cls tower
        for i in range(towernum):
            if i == 0:
                cls_tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz, stride=1, padding=padding,
                                           dilation=dilation))
            else:
                cls_tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1, padding=padding,
                                       dilation=dilation))
            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(non_linear)

        self.add_module('bbox_tower', nn.Sequential(*tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))

        # reg head
        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)
        # cls head
        self.cls_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

    def forward(self, search, kernel, update=None, mode='all'):
        '''mode should be in ['all', 'cls', 'reg']'''
        if mode not in ['cls', 'reg', 'all']:
            raise ValueError("mode must be in ['cls','reg','all']")
        corr_feat = self.pw_corr([kernel], [search])
        if mode == 'cls' or mode == 'all':
            # cls tower + Pred
            c = self.cls_tower(corr_feat)
            cls = 0.1 * self.cls_pred(c)
        if mode == 'reg' or mode == 'all':
            # Reg tower + Pred
            x_reg = self.bbox_tower(corr_feat)
            x = self.adjust * self.bbox_pred(x_reg) + self.bias
            x = torch.exp(x)
        # output
        if mode == 'cls':
            return cls
        elif mode == 'reg':
            return x
        elif mode == 'all':
            return x, cls, None, None

'''2020.09.14 dense connected head module
(1) Reuse previous features
(2) Multiple prediction heads'''
class box_tower_light_dense(nn.Module):
    """
    box tower for FCOS reg
    """
    def __init__(self, inchannels=512, outchannels=256, towernum=1,
                 num_kernel=None, kernel_sz = 3, dilation=1, cat=False,
                 block_name = 'DS', activation='relu'):
        super(box_tower_light_dense, self).__init__()
        '''2020.09.10 support diverse building blocks'''
        if block_name == 'DS':
            block = SeparableConv2d
        elif block_name == 'DSR':
            block = SeparableConv2d_Residual
        elif block_name == 'PDP':
            block = PDP
        else:
            raise ValueError('Block should be DS or DSR')
        if dilation != 1:
            kernel_sz_eq = kernel_sz + (kernel_sz - 1) * (dilation - 1)
            padding = int((kernel_sz_eq - 1) / 2)
        else:
            padding = int((kernel_sz - 1) / 2)
        self.towernum = towernum
        self.bbox_tower = nn.ModuleList()
        self.cls_tower = nn.ModuleList()
        # encode backbone
        self.cls_encode = matrix_light(in_channels=inchannels, out_channels=outchannels, activation=activation)
        self.reg_encode = matrix_light(in_channels=inchannels, out_channels=outchannels, activation=activation)

        self.cls_dw = GroupPW(num_kernel,cat=cat)
        self.reg_dw = GroupPW(num_kernel,cat=cat)
        '''2020.08.25 specify kernel_sz and dilation for towers'''
        if activation == 'relu':
            non_linear = nn.ReLU()
        elif activation == 'hswish':
            non_linear = h_swish()
        else:
            raise ValueError('Unsupported activation function')
        # box pred head
        for i in range(towernum):
            if i == 0:
                if cat:
                    bbox_block_i = block(num_kernel+outchannels, outchannels, kernel_size=kernel_sz,
                                        stride=1, padding=padding, dilation=dilation)  # 8x8=64 channels
                else:
                    # tower.append(nn.Conv2d(num_kernel, outchannels, kernel_size=3, stride=1, padding=1)) # 8x8=64 channels
                    bbox_block_i = block(num_kernel, outchannels, kernel_size=kernel_sz,
                                                 stride=1, padding=padding, dilation=dilation)  # 8x8=64 channels

            else:
                # tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
                if i % 2 == 1:
                    bbox_block_i = block(outchannels+num_kernel, outchannels, kernel_size=kernel_sz, stride=1, padding=padding,
                                       dilation=dilation)
                else:
                    bbox_block_i = block(outchannels, outchannels, kernel_size=kernel_sz, stride=1, padding=padding, dilation=dilation)

            self.bbox_tower.append(nn.Sequential(bbox_block_i, nn.BatchNorm2d(outchannels), non_linear))

        # cls tower
        for i in range(towernum):
            if i == 0:
                if cat:
                    cls_block_i = block(num_kernel+outchannels, outchannels,
                                        kernel_size=kernel_sz, stride=1,
                                        padding=padding, dilation=dilation)
                else:
                    # cls_tower.append(nn.Conv2d(num_kernel, outchannels, kernel_size=3, stride=1, padding=1))
                    cls_block_i = block(num_kernel, outchannels, kernel_size=kernel_sz,
                                        stride=1, padding=padding, dilation=dilation)
            else:
                # cls_tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
                if i % 2 == 1:
                    cls_block_i = block(outchannels+num_kernel, outchannels, kernel_size=kernel_sz,
                                        stride=1, padding=padding, dilation=dilation)

                else:
                    cls_block_i = block(outchannels, outchannels, kernel_size=kernel_sz,
                                        stride=1, padding=padding, dilation=dilation)
            self.cls_tower.append(nn.Sequential(cls_block_i, nn.BatchNorm2d(outchannels), non_linear))


        # reg head & cls head
        for i in range(towernum):
            if i % 2 == 1:
                self.add_module('bbox_pred_%d' % (i + 1),
                                nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1))
                self.add_module('cls_pred_%d' % (i + 1),
                                nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1))
                # adjust scale
                self.__setattr__('adjust_%d'%(i + 1),
                                 nn.Parameter(0.1 * torch.ones(1)))
                self.__setattr__('bias_%d'%(i + 1),
                                 nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda()))


    def forward(self, search, kernel, update=None, mode='all',tracking=False):
        '''mode should be in ['all', 'cls', 'reg']'''
        if mode not in ['cls','reg','all']:
            raise ValueError ("mode must be in ['cls','reg','all']")
        if mode == 'cls' or mode == 'all':
            self.cls_res = []
            # encode first
            if update is None:
                cls_z, cls_x = self.cls_encode(kernel, search)   # [z11, z12, z13]
            else:
                cls_z, cls_x = self.cls_encode(update, search)  # [z11, z12, z13]
            # Point-wise and channel attention
            cls_dw = self.cls_dw(cls_z, cls_x)
            # cls tower
            for i in range(self.towernum):
                if i == 0:
                    x_cls = cls_dw.clone()
                # forward
                x_cls = self.cls_tower[i](x_cls)
                # cat
                if i % 2 == 0:
                    x_cls = torch.cat([x_cls, cls_dw], dim=1)
                # pred
                else:
                    if tracking and i < self.towernum - 1:
                        continue
                    else:
                        self.cls_res.append(0.1 * self.__getattr__('cls_pred_%d'%(i+1))(x_cls))

        if mode == 'reg' or mode == 'all':
            self.reg_res = []
            # encode first
            reg_z, reg_x = self.reg_encode(kernel, search)  # [x11, x12, x13]
            # Point-wise and channel attention
            reg_dw = self.reg_dw(reg_z, reg_x)
            # head
            for i in range(self.towernum):
                if i == 0:
                    x_reg = reg_dw.clone()
                # forward
                x_reg = self.bbox_tower[i](x_reg)
                # cat
                if i % 2 == 0:
                    x_reg = torch.cat([x_reg,reg_dw],dim=1)
                # pred
                else:
                    self.reg_res.append(torch.exp(
                        self.__getattr__('adjust_%d'%(i+1)) * self.__getattr__('bbox_pred_%d'%(i+1))(x_reg)
                        + self.__getattr__('bias_%d'%(i+1))
                    ))

        # output
        if mode == 'cls':
            return self.cls_res
        elif mode == 'reg':
            return self.reg_res
        elif mode == 'all':
            return self.reg_res, self.cls_res

'''2020.09.16 dense connected head module NEW
(1) Reuse previous features (Block2)
(2) Multiple prediction heads (Block4, 6, 8)'''
class box_tower_light_dense_new(nn.Module):
    """
    box tower for FCOS reg
    """
    def __init__(self, inchannels=512, outchannels=256, towernum=1,
                 num_kernel=None, kernel_sz = 3, dilation=1,
                 block_name = 'DS', activation='relu'):
        super(box_tower_light_dense_new, self).__init__()
        '''2020.09.10 support diverse building blocks'''
        if block_name == 'DS':
            block = SeparableConv2d
        elif block_name == 'DSR':
            block = SeparableConv2d_Residual
        elif block_name == 'PDP':
            block = PDP
        else:
            raise ValueError('Block should be DS or DSR')
        if dilation != 1:
            kernel_sz_eq = kernel_sz + (kernel_sz - 1) * (dilation - 1)
            padding = int((kernel_sz_eq - 1) / 2)
        else:
            padding = int((kernel_sz - 1) / 2)
        self.towernum = towernum
        self.bbox_tower = nn.ModuleList()
        self.cls_tower = nn.ModuleList()
        # encode backbone (adjust layer)
        self.cls_encode = matrix_light(in_channels=inchannels, out_channels=outchannels, activation=activation)
        self.reg_encode = matrix_light(in_channels=inchannels, out_channels=outchannels, activation=activation)

        self.cls_dw = GroupPW(num_kernel)
        self.reg_dw = GroupPW(num_kernel)

        self.dim_dict = {3:outchannels, 5:outchannels*2, 7:outchannels*3}
        '''2020.08.25 specify kernel_sz and dilation for towers'''
        if activation == 'relu':
            non_linear = nn.ReLU()
        elif activation == 'hswish':
            non_linear = h_swish()
        else:
            raise ValueError('Unsupported activation function')
        # box pred head
        for i in range(towernum):
            if i == 0:
                dim = outchannels
                bbox_block_i = block(num_kernel, outchannels, kernel_size=kernel_sz,
                                                 stride=1, padding=padding, dilation=dilation)  # 8x8=64 channels

            else:
                # tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
                if i in [4,5]:
                    dim = outchannels * 2
                    bbox_block_i = block(dim, dim, kernel_size=kernel_sz, stride=1, padding=padding,
                                       dilation=dilation)
                elif i in [6,7]:
                    dim = outchannels * 3
                    bbox_block_i = block(dim, dim, kernel_size=kernel_sz, stride=1, padding=padding,
                                         dilation=dilation)
                else:
                    dim = outchannels
                    bbox_block_i = block(outchannels, outchannels, kernel_size=kernel_sz, stride=1, padding=padding, dilation=dilation)

            self.bbox_tower.append(nn.Sequential(bbox_block_i, nn.BatchNorm2d(dim), non_linear))

        # cls tower
        for i in range(towernum):
            if i == 0:
                dim = outchannels
                cls_block_i = block(num_kernel, outchannels, kernel_size=kernel_sz,
                                        stride=1, padding=padding, dilation=dilation)
            else:
                # cls_tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
                if i in [4,5]:
                    dim = outchannels * 2
                    cls_block_i = block(dim, dim, kernel_size=kernel_sz,
                                        stride=1, padding=padding, dilation=dilation)
                elif i in [6,7]:
                    dim = outchannels * 3
                    cls_block_i = block(dim, dim, kernel_size=kernel_sz,
                                        stride=1, padding=padding, dilation=dilation)
                else:
                    dim = outchannels
                    cls_block_i = block(outchannels, outchannels, kernel_size=kernel_sz,
                                        stride=1, padding=padding, dilation=dilation)
            self.cls_tower.append(nn.Sequential(cls_block_i, nn.BatchNorm2d(dim), non_linear))


        # reg head & cls head
        for i in range(towernum):
            if i in [3,5,7]:
                self.add_module('bbox_pred_%d' % (i + 1),
                                nn.Conv2d(self.dim_dict[i], 4, kernel_size=3, stride=1, padding=1))
                self.add_module('cls_pred_%d' % (i + 1),
                                nn.Conv2d(self.dim_dict[i], 1, kernel_size=3, stride=1, padding=1))
                # adjust scale
                self.__setattr__('adjust_%d'%(i + 1),
                                 nn.Parameter(0.1 * torch.ones(1)))
                self.__setattr__('bias_%d'%(i + 1),
                                 nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda()))


    def forward(self, search, kernel, update=None, mode='all',tracking=False):
        '''mode should be in ['all', 'cls', 'reg']'''
        if mode not in ['cls','reg','all']:
            raise ValueError ("mode must be in ['cls','reg','all']")
        if mode == 'cls' or mode == 'all':
            self.cls_res = []
            # encode first
            if update is None:
                cls_z, cls_x = self.cls_encode(kernel, search)   # [z11, z12, z13]
            else:
                cls_z, cls_x = self.cls_encode(update, search)  # [z11, z12, z13]
            # Point-wise and channel attention
            cls_dw = self.cls_dw(cls_z, cls_x)
            # cls tower
            for i in range(self.towernum):
                if i == 0:
                    x_cls = cls_dw.clone()
                # forward
                x_cls = self.cls_tower[i](x_cls)
                if i == 1:
                    base_cls = x_cls.clone()
                # predict
                if i in [3, 5, 7]:
                    self.cls_res.append(0.1 * self.__getattr__('cls_pred_%d' % (i + 1))(x_cls))
                # cat
                if i in [3,5]:
                    x_cls = torch.cat([x_cls, base_cls], dim=1)

                if tracking and i < self.towernum - 1:
                    continue


        if mode == 'reg' or mode == 'all':
            self.reg_res = []
            # encode first
            reg_z, reg_x = self.reg_encode(kernel, search)  # [x11, x12, x13]
            # Point-wise and channel attention
            reg_dw = self.reg_dw(reg_z, reg_x)
            # head
            for i in range(self.towernum):
                if i == 0:
                    x_reg = reg_dw.clone()
                # forward
                x_reg = self.bbox_tower[i](x_reg)
                if i == 1:
                    base_reg = x_reg.clone()
                # predict
                if i in [3,5,7]:
                    self.reg_res.append(torch.exp(
                        self.__getattr__('adjust_%d'%(i+1)) * self.__getattr__('bbox_pred_%d'%(i+1))(x_reg)
                        + self.__getattr__('bias_%d'%(i+1))
                    ))
                # cat
                if i in [3,5]:
                    x_reg = torch.cat([x_reg,base_reg],dim=1)
                # pred
                if tracking and i < self.towernum - 1:
                    continue



        # output
        if mode == 'cls':
            return self.cls_res
        elif mode == 'reg':
            return self.reg_res
        elif mode == 'all':
            return self.reg_res, self.cls_res



'''2020.10.19 Use normal conv rather than DS conv'''
class box_tower_large(nn.Module):
    """
    box tower for FCOS reg
    """
    def __init__(self, outchannels=256, towernum=1,
                 num_kernel=None, kernel_sz = 3, dilation=1, cat=False,
                 CA=True, matrix=False, BN_choice=None, inchannels=112,
                 bbox_compute='exp', stride=16):
        super(box_tower_large, self).__init__()

        block = nn.Conv2d
        non_linear = nn.ReLU()
        self.bbox_compute = bbox_compute
        self.stride = stride

        if dilation != 1:
            kernel_sz_eq = kernel_sz + (kernel_sz - 1) * (dilation - 1)
            padding = int((kernel_sz_eq - 1) / 2)
        else:
            padding = int((kernel_sz - 1) / 2)
        tower = []
        cls_tower = []
        '''BN before Pointwise Corr'''
        self.BN_choice = BN_choice
        if self.BN_choice == 'before':
            '''template and search use separate BN'''
            self.BN_adj_z = nn.BatchNorm2d(inchannels)
            self.BN_adj_x = nn.BatchNorm2d(inchannels)
        '''Point-wise Correlation'''
        self.pw_corr = GroupPW(num_kernel, cat=cat, CA=CA, matrix=matrix)

        # box pred head
        for i in range(towernum):
            if i == 0:
                tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz,
                                       stride=1, padding=padding, dilation=dilation))  # 8x8=64 channels

            else:
                tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1,
                                   padding=padding, dilation=dilation))

            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(non_linear)

        # cls tower
        for i in range(towernum):
            if i == 0:
                cls_tower.append(block(num_kernel, outchannels, kernel_size=kernel_sz, stride=1, padding=padding, dilation=dilation))
            else:
                cls_tower.append(block(outchannels, outchannels, kernel_size=kernel_sz, stride=1, padding=padding, dilation=dilation))

            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(non_linear)

        self.add_module('bbox_tower', nn.Sequential(*tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))


        # reg head
        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

        if bbox_compute == 'exp':
            # adjust scale
            self.adjust = nn.Parameter(0.1 * torch.ones(1))
            self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())
        elif bbox_compute == 'linear':
            pass
        else:
            raise ValueError ('Unsupported bbox compute method')

    def forward(self, search, kernel, update=None, mode='all'):
        '''mode should be in ['all', 'cls', 'reg']'''
        if mode not in ['cls','reg','all']:
            raise ValueError ("mode must be in ['cls','reg','all']")
        if self.BN_choice == 'before':
            kernel, search = self.BN_adj_z(kernel), self.BN_adj_x(search)
        corr_feat = self.pw_corr([kernel], [search])
        if mode == 'cls' or mode == 'all':
            # cls tower + Pred
            c = self.cls_tower(corr_feat)
            cls = 0.1 * self.cls_pred(c)
        if mode == 'reg' or mode == 'all':
            # Reg tower + Pred
            x_reg = self.bbox_tower(corr_feat)
            if self.bbox_compute == 'exp':
                x = self.adjust * self.bbox_pred(x_reg) + self.bias
                x = torch.exp(x)
            elif self.bbox_compute == 'linear':
                x = self.stride * nn.functional.relu(self.bbox_pred(x_reg))
            else:
                raise ValueError('Unsupported bbox compute method')
        # output
        if mode == 'cls':
            return cls
        elif mode == 'reg':
            return x
        elif mode == 'all':
            return x, cls, None, None
