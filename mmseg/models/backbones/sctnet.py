import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Conv2d,ConvModule)
from mmcv.runner import BaseModule
from ..builder import BACKBONES
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,trunc_normal_init,normal_init)
from timm.models.layers import DropPath

@BACKBONES.register_module()
class SCTNet(BaseModule):
    """
    The SCTNet implementation based on mmSegmentation.
    Args:
        layer_nums (List, optional): The layer nums of every stage. Default: [2, 2, 2, 2]
        base_channels (int, optional): The base channels. Default: 64
        spp_channels (int, optional): The channels of DAPPM. Defualt: 128
        in_channels (int, optional): The channels of input image. Default: 3
        num_heads (int, optional): The num of heads in CFBlock. Default: 8
        drop_rate (float, optional): The drop rate in CFBlock. Default:0.
        drop_path_rate (float, optional): The drop path rate in CFBlock. Default: 0.2
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 layer_nums=[2, 2, 2, 2],
                 base_channels=64,  #Slim32  
                 spp_channels=128,  #Slim64
                 in_channels=3,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 pretrained=None,
                 init_cfg=None):
        super(SCTNet,self).__init__(init_cfg=init_cfg)
        self.base_channels = base_channels
        base_chs = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, base_chs, kernel_size=3, stride=2, padding=1),
            nn.SyncBatchNorm(base_chs),
            nn.ReLU(),
            nn.Conv2d(
                base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            nn.SyncBatchNorm(base_chs),
            nn.ReLU(), )
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs,
                                       layer_nums[0])
        self.layer2 = self._make_layer(
            BasicBlock, base_chs, base_chs * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, base_chs * 2, base_chs * 4, layer_nums[2], stride=2)

        self.layer3_2 = CFBlock(
            in_channels=base_chs * 4,
            out_channels=base_chs * 4,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate)

        
        self.convdown4 = nn.Sequential(
            nn.Conv2d(
                base_chs*4, base_chs*8, kernel_size=3, stride=2, padding=1),
            nn.SyncBatchNorm(base_chs*8),
            nn.ReLU(),)
        self.layer4 = CFBlock(
            in_channels=base_chs * 8,
            out_channels=base_chs * 8,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate)
        self.layer5 = CFBlock(
            in_channels=base_chs * 8,
            out_channels=base_chs * 8,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate)


        self.spp = DAPPM_head(
            base_chs * 8, spp_channels, base_chs * 2)

        if self.init_cfg.type == 'Pretrained':
            super(SCTNet, self).init_weights()
        else:
            self.init_weight()

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def init_weight(self):
        self.conv1.apply(self._init_weights_kaiming)
        self.layer1.apply(self._init_weights_kaiming)
        self.layer2.apply(self._init_weights_kaiming)
        self.layer3.apply(self._init_weights_kaiming)
        self.convdown4.apply(self._init_weights_kaiming)
        self.spp.apply(self._init_weights_kaiming)


    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride),
                nn.SyncBatchNorm(out_channels))

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(self.conv1(x))  # c, 1/4
        x2 = self.layer2(self.relu(x1))  # 2c, 1/8
        x3_1 = self.layer3(self.relu(x2))  # 4c, 1/16
        x3 = self.layer3_2(self.relu(x3_1))  # 4c, 1/16
        x4_down=self.convdown4(x3)  
        x4 = self.layer4(self.relu(x4_down))  # 8c, 1/32
        x5 = self.layer5(self.relu(x4))  # 8c, 1/32
        x6 = self.spp(x5)   # 2c, 1/32
        x7 = F.interpolate(
            x6, size=x2.shape[2:], mode='bilinear')  # 2c, 1/8
        x_out = torch.cat([x2, x7], dim=1)  # 4c, 1/8
        logit_list = [x_out, x2,[x,[x_out,x5,x3]]] 

        return logit_list


#conv->bn->relu->conv->bn->relu
class BasicBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super(BasicBlock,self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.SyncBatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.SyncBatchNorm(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out if self.no_relu else self.relu(out)


#BN->Conv->GELU->drop->Conv2->drop
class MLP(BaseModule):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP,self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.SyncBatchNorm(in_channels, eps=1e-06)  #TODO,1e-6?
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class ConvolutionalAttention(BaseModule):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8):
        super(ConvolutionalAttention,self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv =nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 =nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_init(self.kv, std=0.001)
        trunc_normal_init(self.kv3, std=0.001)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.)
            constant_init(m.bias, val=.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)


    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,c_inter,h,w -> n,heads,c_inner//heads,hw
        x = F.softmax(x, dim=3)   
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)  
        x = x.reshape([x_shape[0], self.inter_channels, h, w]) 
        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor. (n,c,h,w)
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        x1 = F.conv2d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=(3,0))  
        x1 = self._act_dn(x1)  
        x1 = F.conv2d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=(3,0))  
        x3 = F.conv2d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=(0,3)) 
        x3 = self._act_dn(x3)
        x3 = F.conv2d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=(0,3)) 
        x=x1+x3
        return x

class CFBlock(BaseModule):
    """
    The CFBlock implementation based on PaddlePaddle.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in CFBlock. Default: 0.2
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.):
        super(CFBlock,self).__init__()
        in_channels_l = in_channels
        out_channels_l = out_channels
        self.attn_l = ConvolutionalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn_l(x))
        x = x + self.drop_path(self.mlp_l(x)) 
        return x



class DAPPM_head(BaseModule):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(DAPPM_head,self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=5, stride=2, padding=2, count_include_pad =True),
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=9, stride=4, padding=4, count_include_pad =True),
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=17, stride=8, padding=8, count_include_pad =True),
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale0 = nn.Sequential(
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.process1 = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels),
            nn.ReLU(),
            Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process2 = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels),
            nn.ReLU(),
            Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process3 = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels),
            nn.ReLU(),
            Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process4 = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels),
            nn.ReLU(),
            Conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.compression = nn.Sequential(
            nn.SyncBatchNorm(
                inter_channels * 5),
            nn.ReLU(),
            Conv2d(
                inter_channels * 5,
                out_channels,
                kernel_size=1))
        self.shortcut = nn.Sequential(
            nn.SyncBatchNorm(
                in_channels),
            nn.ReLU(),
            Conv2d(
                in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x_shape = x.shape[2:]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1((F.interpolate(
                self.scale1(x), size=x_shape, mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(
            self.scale2(x), size=x_shape, mode='bilinear') + x_list[1]))))
        x_list.append(
            self.process3((F.interpolate(
                self.scale3(x), size=x_shape, mode='bilinear') + x_list[2])))
        x_list.append(
            self.process4((F.interpolate(
                self.scale4(x), size=x_shape, mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)
        return out
