import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from torch import einsum

from itertools import repeat
from functools import partial
from typing import Union, List, Tuple, Optional, Callable
import numpy as np
import math

from lib.models.activations import *

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)

class AveragePooler(nn.Module):

    def __init__(self, seq_red, c_dim=256, hidden_dim=128) -> None:
        super(AveragePooler, self).__init__()

        '''
        Module to reduce the sequence size

        Args:
            - seq_red: How much to reduce 
            - c_dim: channel dimension 
            - hidden_dim: hidden dimension
        '''

        self.global_pooling = nn.AdaptiveAvgPool2d(seq_red)
        self.flatten = nn.Flatten(start_dim = 2)
        self.fc1 = nn.Linear(c_dim, hidden_dim)
        self.act = nn.ReLU(inplace=False)
        

    def forward(self, x):
        
        x = self.global_pooling(x)
        x = self.flatten(x).permute(0,2,1)
        x = self.fc1(x)
        x = self.act(x)
        
        return x 

class ExemplarTransformer(nn.Module):

    def __init__(self, in_channels, 
                       out_channels, 
                       dw_padding, 
                       pw_padding=0,
                       dw_stride=1, 
                       pw_stride=1,
                       e_exemplars=4, 
                       temperature=30, 
                       hidden_dim=256, 
                       dw_kernel_size=5, 
                       pw_kernel_size=1,
                       layer_norm_eps = 1e-05,
                       dim_feedforward = 1024, # 2048,
                       ff_dropout = 0.1,
                       ff_activation = "relu",
                       num_heads = 8,
                       seq_red = 1,
                       se_ratio = 0.5,
                       se_kwargs = None,
                       se_act_layer = "relu",
                       norm_layer = nn.BatchNorm2d,
                       norm_kwargs = None,
                       sm_normalization = False,
                       dropout = False,
                       dropout_rate = 0.1) -> None:
        super(ExemplarTransformer, self).__init__()

        '''
        
        Sub Models:
            - average_pooler: attention module
            - K (keys): Representing the last layer of the average pooler 
                        K is used for the computation of the mixing weights.
                        The mixing weights are used for the both the spatial as well as the
                        pointwise convolution
            - V (values): Representing the different kernels. 
                          There have to be two sets of values, one for the spatial and one for the pointwise 
                          convolution. The shape of the kernels differ. 


        Args:
            - in_channels: number of input channels
            - out_channels: number of output channels
            - padding: input padding for when applying kernel
            - stride: stride for kernel application
            - e_exemplars: number of expert kernels
            - temperature: temperature for softmax
            - hidden_dim: hidden dimension used in the average pooler
            - kernel_size: kernel size used for the weight shape computation
            - layernorm eps: used for layer norm after the convolution operation
            - dim_feedforward: dimension for FF network after attention module,
            - ff_dropout: dropout rate for FF network after attention module
            - activation: activation function for FF network after attention module
            - num_heads: number of heads
            - seq_red: sequence reduction dimension for the global average pooling operation


        '''

        ## general parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.e_exemplars = e_exemplars
        norm_kwargs = norm_kwargs or {}
        self.hidden_dim = hidden_dim
        self.sm_norm = sm_normalization
        self.K = nn.Parameter(torch.randn(e_exemplars, hidden_dim)) # could be an embedding / a mapping from X to K instead of pre-learned
        self.dropout = dropout
        self.do = nn.Dropout(dropout_rate)

        ## average pool 
        self.temperature = temperature
        self.average_pooler = AveragePooler(seq_red=seq_red, c_dim=in_channels, hidden_dim=hidden_dim) #.cuda()
        self.softmax = nn.Softmax(dim=-1)
        
        ## multihead setting
        self.H = num_heads
        self.head_dim = self.hidden_dim // self.H

        ## depthwise convolution parameters
        self.dw_groups = self.out_channels
        self.dw_kernel_size = _pair(dw_kernel_size)
        self.dw_padding = dw_padding
        self.dw_stride = dw_stride
        self.dw_weight_shape = (self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size
        dw_weight_num_param = 1
        for wd in self.dw_weight_shape:
            dw_weight_num_param *= wd
        self.V_dw = nn.Parameter(torch.Tensor(e_exemplars, dw_weight_num_param))
        self.dw_bn = norm_layer(self.in_channels, **norm_kwargs)
        self.dw_act = nn.ReLU(inplace=True)

        ## pointwise convolution parameters
        self.pw_groups = 1
        self.pw_kernel_size = _pair(pw_kernel_size)
        self.pw_padding = pw_padding
        self.pw_stride = pw_stride
        self.pw_weight_shape = (self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size
        pw_weight_num_param = 1
        for wd in self.pw_weight_shape:
            pw_weight_num_param *= wd
        self.V_pw = nn.Parameter(torch.Tensor(e_exemplars, pw_weight_num_param))
        self.pw_bn = norm_layer(self.out_channels, **norm_kwargs)
        self.pw_act = nn.ReLU(inplace=False)

        ## Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            se_kwargs = resolve_se_args(se_kwargs, self.in_channels, nn.ReLU) #_get_activation_fn(se_act_layer))
            self.se = SqueezeExcite(self.in_channels, se_ratio=se_ratio, **se_kwargs)
        
        ## Implementation of Feedforward model after the QKV part
        self.linear1 = nn.Linear(self.out_channels, dim_feedforward)
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, self.out_channels)
        self.norm1 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.out_channels, eps=layer_norm_eps)
        self.ff_dropout1 = nn.Dropout(ff_dropout)
        self.ff_dropout2 = nn.Dropout(ff_dropout)
        self.ff_activation = _get_activation_fn(ff_activation)

        # initialize the kernels 
        self.reset_parameters()

    def reset_parameters(self):
        init_weight_dw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.dw_weight_shape)
        init_weight_dw(self.V_dw)

        init_weight_pw = get_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.e_exemplars, self.pw_weight_shape)
        init_weight_pw(self.V_pw)
    


    def forward(self, x):

        residual = x
        
        # X: [B,C,H,W]

        # apply average pooler
        q = self.average_pooler(x)
        d_k = q.shape[-1]
        # Q: [B,S,C]

        # outer product with keys
        #qk = einsum('b n c, k c -> b n k', q, self.K) # K^T: [C, K] QK^T: [B,S,K]
        qk = torch.matmul(q, self.K.T)
        
        if self.sm_norm:
            qk = 1/math.sqrt(d_k) * qk

        # apply softmax 
        attn = self.softmax(qk/self.temperature) # -> [batch_size, e_exemplars]
        
        # multiply attention map with values 
        #dw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_dw) # V: [K, E_dw]
        #pw_qkv_kernel = einsum('b s k, k e -> b s e', attn, self.V_pw) # V: [K, E_pw]
        dw_qkv_kernel = torch.matmul(attn, self.V_dw) # V: [K, E_dw]
        pw_qkv_kernel = torch.matmul(attn, self.V_pw) # V: [K, E_pw]

        ###########################################################################################
        ####### convolve input with the output instead of adding it to it in a residual way #######
        ###########################################################################################

        ## dw conv
        B, C, H, W = x.shape

        # dw conv
        dw_weight_shape = (B * self.out_channels, self.in_channels // self.dw_groups) + self.dw_kernel_size
        dw_weight = dw_qkv_kernel.view(dw_weight_shape)
        
        # reshape the input
        x = x.reshape(1, B * C, H, W)
        
        # apply convolution
        x = F.conv2d(
            x, dw_weight, bias=None, stride=self.dw_stride, padding=self.dw_padding, 
            groups=self.dw_groups * B)
        
        x = x.permute([1, 0, 2, 3]).view(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = self.dw_bn(x)
        x = self.dw_act(x)

        ## SE
        x = self.se(x)

        ## pw conv
        B, C, H, W = x.shape

        # dw conv
        pw_weight_shape = (B * self.out_channels, self.in_channels // self.pw_groups) + self.pw_kernel_size
        pw_weight = pw_qkv_kernel.view(pw_weight_shape)
        
        # reshape the input
        x = x.view(1, B * C, H, W)
        
        # apply convolution
        x = F.conv2d(
            x, pw_weight, bias=None, stride=self.pw_stride, padding=self.pw_padding, 
            groups=self.pw_groups * B)
        
        x = x.permute([1, 0, 2, 3]).view(B, self.out_channels, x.shape[-2], x.shape[-1])
        x = self.pw_bn(x)
        x = self.pw_act(x)


        if self.dropout:
            x = x + self.do(residual)
        else:
            x = x + residual

        
        # reshape output of convolution operation
        out = x.view(B, self.out_channels, -1).permute(0,2,1)
        
        # FF network 
        out = self.norm1(out)
        out2 = self.linear2(self.ff_dropout(self.ff_activation(self.linear1(out))))
        out = out + self.ff_dropout2(out2)
        out = self.norm2(out)
        out = out.permute(0,2,1).view(B,C,H,W)
        
        return out
    
def get_initializer(initializer, e_exemplars, expert_shape):
        def initializer_func(weight):
            """Initializer function."""
            num_params = np.prod(expert_shape)
            if (len(weight.shape) != 2 or weight.shape[0] != e_exemplars or weight.shape[1] != num_params):
                raise (ValueError('Variables must have shape [e_exemplars, num_params]'))
            for i in range(e_exemplars):
                initializer(weight[i].view(expert_shape))
        return initializer_func

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "se_relu":
        return nn.ReLU

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
        
_SE_ARGS_DEFAULT = dict(
    gate_fn=sigmoid,
    act_layer=None,  # None == use containing block's activation layer
    reduce_mid=False,
    divisor=1)

def resolve_se_args(kwargs, in_chs, act_layer=None):
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    # fill in args that aren't specified with the defaults
    for k, v in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, v)
    # some models, like MobilNetV3, calculate SE reduction chs from the containing block's mid_ch instead of in_ch
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    # act_layer override, if it remains None, the containing block's act_layer will be used
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs

def make_divisible(v: int, divisor: int = 8, min_value: int = None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # ensure round down does not go down by more than 10%.
        new_v += divisor
    return new_v

class SqueezeExcite(nn.Module):

    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1):
        super(SqueezeExcite, self).__init__()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

