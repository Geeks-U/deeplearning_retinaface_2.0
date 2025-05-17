import copy

import torch
import torch.nn as nn

# 注册表
FPN_REGISTRY = {}
# 装饰器
def register_fpn(name:str):
    def decorator(cls):
        FPN_REGISTRY[name] = cls
        return cls
    return decorator


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

cfg_fpn_default = {
    'in_channels': [32, 64, 160],
    'out_names': ['high', 'mid', 'low'],
    'out_steps': [1, 1, 1],
    'out_channels': [64, 64, 64]
}

@register_fpn('fpn')
class FPN(nn.Module):
    def __init__(self, cfg_fpn=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg_fpn_default)
        if cfg_fpn is not None:
            self.cfg.update(cfg_fpn)

        self.stage_high = FPNStageHigh(in_channels=self.cfg['in_channels'][0], out_channels=self.cfg['out_channels'][0])
        self.stage_mid = FPNStageMid(in_channels=self.cfg['in_channels'][1], out_channels=self.cfg['out_channels'][1])
        self.stage_low = FPNStageLow(in_channels=self.cfg['in_channels'][2], out_channels=self.cfg['out_channels'][2])

    def forward(self, x):
        output_stage_low = self.stage_low(x['low'])
        output_stage_mid = self.stage_mid([x['mid'], output_stage_low])
        output_stage_high = self.stage_high([x['high'], output_stage_mid])
        return {
            self.cfg['out_names'][0]: output_stage_high,
            self.cfg['out_names'][1]: output_stage_mid,
            self.cfg['out_names'][2]: output_stage_low
        }


# 工厂函数
def build_fpn(model_name: str, cfg_fpn=None) -> nn.Module:
    if model_name not in FPN_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    return FPN_REGISTRY[model_name](cfg_fpn=cfg_fpn)


from src.nets.backbone import build_backbone

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)

    mn = build_backbone('mobilenetv2', cfg_backbone={'pretrained': True})
    mn.eval()

    with torch.no_grad():
        outputs = mn(x)
    print("mn模型配置:", mn.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={mn.cfg['out_steps'][i]})")

    fpn = build_fpn('fpn', cfg_fpn={'out_steps': [1, 1, 1]})
    fpn.eval()

    with torch.no_grad():
        outputs = fpn(outputs)
    print("fpn模型配置:", fpn.cfg)

    print("各输出特征图形状：")
    for i, name in enumerate(['high', 'mid', 'low']):
        feat = outputs[name]
        print(f"{name}: {feat.shape} (stride={fpn.cfg['out_steps'][i]})")
