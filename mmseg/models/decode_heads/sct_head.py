# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (Conv2d)
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,trunc_normal_init)

#bn-conv-bn-conv
@HEADS.register_module()
class SCTHead(BaseDecodeHead):
    def __init__(self,
                **kwargs):
        super(SCTHead,self).__init__(**kwargs)
        self.bn1 = nn.SyncBatchNorm(self.in_channels)
        self.conv1 = Conv2d(
            self.in_channels,
            self.channels,
            kernel_size=3,
            padding=1)
        self.bn2 = nn.SyncBatchNorm(self.channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.cls_seg(self.relu(self.bn2(x)))
        return x,out
    
    def init_weights(self):
        for m in self.modules():
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
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        decoder_feature,seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses,decoder_feature,seg_logits
            
    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        x,out = self.forward(inputs)
        return out