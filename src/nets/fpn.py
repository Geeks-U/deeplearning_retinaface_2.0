# 框架导入
import torch
import torch.nn as nn
import copy


class ConvBNLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, negative_slope=0.01):
        super().__init__()
        self.block = torch.nn.Sequential(
            nn.modules.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.modules.BatchNorm2d(out_channels),
            nn.modules.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class FPNStageHigh(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block0 = ConvBNLeakyReLU(in_channels=in_channels, out_channels=out_channels)
        self.up = nn.modules.Upsample(scale_factor=2, mode='nearest')
        self.block1 = ConvBNLeakyReLU(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        output_block0 = self.block0(x[0])
        output_block1 = self.block1(output_block0 + self.up(x[1]))
        return output_block1

class FPNStageMid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block0 = ConvBNLeakyReLU(in_channels=in_channels, out_channels=out_channels)
        self.up = nn.modules.Upsample(scale_factor=2, mode='nearest')
        self.block1 = ConvBNLeakyReLU(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        output_block0 = self.block0(x[0])
        output_block1 = self.block1(output_block0 + self.up(x[1]))
        return output_block1

class FPNStageLow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block0 = ConvBNLeakyReLU(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        output_block0 = self.block0(x)
        return output_block0

cfg_default = {
    'in_channels': [32, 64, 160],
    'out_channels': [64, 64, 64],
    'out_step': [1, 1, 1]
}

class FPN(nn.Module):
    def __init__(self, cfg_fpn=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_default)
        if cfg_fpn is not None:
            self.cfg.update(cfg_fpn)

        self.stage_high = FPNStageHigh(in_channels=self.cfg['in_channels'][0], out_channels=self.cfg['out_channels'][0])
        self.stage_mid = FPNStageMid(in_channels=self.cfg['in_channels'][1], out_channels=self.cfg['out_channels'][1])
        self.stage_low = FPNStageLow(in_channels=self.cfg['in_channels'][2], out_channels=self.cfg['out_channels'][2])

    @property
    def out_steps(self):
        return self.cfg['out_step']

    def forward(self, x):
        output_stage_low = self.stage_low(x['low'])
        output_stage_mid = self.stage_mid([x['mid'], output_stage_low])
        output_stage_high = self.stage_high([x['high'], output_stage_mid])
        return {
            'high': output_stage_high,
            'mid': output_stage_mid,
            'low': output_stage_low
        }

from src.nets.backbone import MobileNetV2

if __name__ == '__main__':
    mn = MobileNetV2()
    mn.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = mn(x)
    print("mn模型配置:", mn.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={mn.out_steps[i]})")

    fpn = FPN()
    fpn.eval()

    with torch.no_grad():
        outputs = fpn(outputs)
    print("fpn模型配置:", fpn.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={fpn.out_steps[i]})")
