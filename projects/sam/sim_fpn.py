# From https://github.com/open-mmlab/mmdetection/pull/9812/files
from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType
from mmengine.model import BaseModule
from torch import Tensor


@MODELS.register_module()
class SimpleFPN(BaseModule):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(self,
                 backbone_channel: int,
                 in_fpn_level: Tuple[int] = (0, 3),
                 in_channels: List[int] = [192, 384, 768, 768],
                 out_channels: int = 256,
                 num_outs: int = 5,
                 freeze_levels: List[int] = [2, ],
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_fpn_level = in_fpn_level
        self.identy_level = 2
        if in_fpn_level[1] < self.identy_level or in_fpn_level[0] > self.identy_level:
            self.identy_level = None
        else:
            self.identy_level = self.identy_level - in_fpn_level[0]
        
        self.num_ins = in_fpn_level[1] - in_fpn_level[0] + 1
        assert self.num_ins == len(in_channels)
        self.num_outs = num_outs

        self.fpns = nn.ModuleList()

        # 1/4
        self.fpns.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.backbone_channel,
                                self.backbone_channel // 2, 2, 2),
                build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
                nn.GELU(),
                nn.ConvTranspose2d(self.backbone_channel // 2,
                                self.backbone_channel // 4, 2, 2)
            )
        )


        # 1/8
        self.fpns.append(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2)
        )
        # 1/16
        self.fpns.append(nn.Identity())
        # 1/32
        self.fpns.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.fpns = self.fpns[in_fpn_level[0]:in_fpn_level[1] + 1]

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i in freeze_levels:
                l_conv = l_conv.eval()
                fpn_conv = fpn_conv.eval()
                for param in l_conv.parameters():
                    param.requires_grad = False
                for param in fpn_conv.parameters():
                    param.requires_grad = False

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        for fpn in self.fpns:
            inputs.append(fpn(input))
        # inputs.append(self.fpn1(input))
        # inputs.append(self.fpn2(input))
        # inputs.append(self.fpn3(input))
        # inputs.append(self.fpn4(input))

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)
